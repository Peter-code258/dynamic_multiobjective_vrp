"""事件驱动滚动重规划模拟器。

约定：
- 事件时刻先推进车辆到“已完成步的边界”，再更新业务状态；
- 已服务客户、已行驶距离和响应时长锁定，重规划只安排剩余客户；
- 故障车辆立即不可用，未完成客户回到待规划集合；
- 拥堵只改变行驶时间，不改变距离；
- 无法服务时保留未服务客户并写入结果，不静默丢弃。
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .algorithms import get_planner
from .algorithms.base import AlgorithmResult, PlannerConfig
from .baselines import run_greedy
from .context import PlanningContext
from .models import (
    ACTIVE,
    ASSIGNED,
    BROKEN,
    Customer,
    FINISHED,
    IDLE,
    PENDING,
    Scenario,
    SERVED,
    Solution,
    UNRELEASED,
    UNSERVED,
    VRPInstance,
)
from .travel_time import TravelNetwork


EPS = 1e-9


@dataclass
class SimulationConfig:
    horizon: float = 600.0
    speed: float = 1.0
    max_route_duration: Optional[float] = None
    hard_time_windows: bool = False
    selection_rule: str = "ideal_point"


def _stable_algorithm_seed(seed: int, algorithm: str) -> int:
    code = sum((idx + 1) * ord(ch) for idx, ch in enumerate(algorithm))
    return int((int(seed) * 1000003 + code * 97) % (2 ** 32 - 1))


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def select_compromise(solutions: Sequence[Solution]) -> Optional[Solution]:
    """归一化理想点规则：所有算法与事件使用同一折中解选择方式。"""
    if not solutions:
        return None
    vectors = np.array([s.objectives for s in solutions], dtype=float)
    lo = vectors.min(axis=0)
    hi = vectors.max(axis=0)
    span = np.where(hi - lo > EPS, hi - lo, 1.0)
    normalized = (vectors - lo) / span
    scores = np.sum(normalized ** 2, axis=1)
    best = int(np.argmin(scores))
    return solutions[best]


class SimulationEngine:
    def __init__(
        self,
        instance: VRPInstance,
        scenario: Scenario,
        algorithm: str,
        seed: int,
        planner_config: PlannerConfig,
        sim_config: Optional[SimulationConfig] = None,
    ) -> None:
        self.instance = instance.depth_copy()
        self.scenario = scenario
        self.algorithm = algorithm
        self.seed = int(seed)
        self.planner_config = planner_config
        self.config = sim_config or SimulationConfig()
        self.travel = TravelNetwork(self.instance, speed=self.config.speed)
        self.rng = np.random.default_rng(_stable_algorithm_seed(self.seed, algorithm))

        self.clock = 0.0
        self.used_vehicle_ids: Set[int] = set()
        self.replans: List[Dict[str, object]] = []
        self.recovery_records: List[Dict[str, object]] = []
        self.pending_recoveries: List[Dict[str, object]] = []
        self.evaluations_elapsed = 0
        self.network_version = 0
        self.last_plan: Dict[int, List[int]] = {}
        self.last_archive: List[Solution] = []
        self.archive_history: List[Dict[str, object]] = []
        self.had_infeasible_plan = False
        self.dynamic_customers: Dict[int, Customer] = {}
        for event in scenario.events:
            if event.event_type == "NEW_ORDER":
                raw = event.payload.get("customer", {})
                cid = int(raw["customer_id"])
                self.dynamic_customers[cid] = Customer(
                    customer_id=cid,
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                    demand=float(raw["demand"]),
                    service_time=float(raw.get("service_time", 0.0)),
                    release_time=float(raw.get("release_time", event.event_time)),
                    ready_time=float(raw.get("ready_time", raw.get("release_time", event.event_time))),
                    due_time=None if raw.get("due_time") is None else float(raw["due_time"]),
                    status=UNRELEASED,
                )

    # ------------------------------------------------------------------
    # 状态推进
    # ------------------------------------------------------------------
    def _advance_to(self, target_time: float) -> None:
        """推进车辆执行已计划路线，直到下一步开始时刻不早于目标时刻。

        为避免把正在执行的弧段或服务拆成不可审计的半状态，只要车辆在目标时刻
        之前已经出发，就完整执行该“行驶 + 服务”步，然后停在步边界。因此事件
        的实际规划时刻可能略晚于事件名义时刻，记录为 planning_time。
        """
        for _ in range(100000):
            progressed = False
            for vid in sorted(self.instance.vehicles.keys()):
                vehicle = self.instance.vehicles[vid]
                if vehicle.status != ACTIVE:
                    continue
                if not vehicle.route_suffix:
                    vehicle.status = IDLE
                    continue
                if vehicle.current_time >= target_time - EPS:
                    continue
                cid = vehicle.route_suffix[0]
                customer = self.instance.customers.get(cid)
                if customer is None:
                    vehicle.route_suffix.pop(0)
                    progressed = True
                    continue
                if customer.status == SERVED:
                    vehicle.route_suffix.pop(0)
                    progressed = True
                    continue
                departure = vehicle.current_time
                arrival = departure + self.travel.travel_time(
                    vehicle.current_node, cid, departure
                )
                start = max(
                    arrival,
                    float(customer.release_time),
                    float(customer.ready_time),
                )
                completion = start + float(customer.service_time)
                vehicle.cumulative_distance += self.travel.distance(
                    vehicle.current_node, cid
                )
                vehicle.cumulative_response += start - float(customer.release_time)
                vehicle.cumulative_demand += float(customer.demand)
                vehicle.current_node = cid
                vehicle.current_time = completion
                vehicle.served_customers.append(cid)
                vehicle.route_suffix.pop(0)
                vehicle.used = True
                self.used_vehicle_ids.add(vid)
                customer.status = SERVED
                customer.service_start_time = start
                customer.arrival_time = arrival
                customer.assigned_vehicle_id = vid
                if not vehicle.route_suffix:
                    vehicle.status = IDLE
                progressed = True
            if not progressed:
                break

    def _release_due(self, now: float) -> None:
        for customer in self.instance.customers.values():
            if customer.status == UNRELEASED and customer.release_time <= now + EPS:
                customer.status = PENDING

    def _apply_event(self, event) -> Dict[str, object]:
        """应用事件并返回受影响对象，供重规划记录与恢复指标使用。"""
        if event.event_type == "NEW_ORDER":
            raw = event.payload.get("customer", {})
            cid = int(raw["customer_id"])
            if cid in self.instance.customers:
                return {"customers": []}
            template = self.dynamic_customers.get(cid)
            if template is None:
                template = Customer(
                    customer_id=cid,
                    x=float(raw["x"]),
                    y=float(raw["y"]),
                    demand=float(raw["demand"]),
                    service_time=float(raw.get("service_time", 0.0)),
                    release_time=float(raw.get("release_time", event.event_time)),
                )
            customer = template.copy()
            customer.status = PENDING
            customer.release_time = float(raw.get("release_time", customer.release_time))
            self.instance.add_customer(customer)
            self.travel.add_node(cid, customer.x, customer.y)
            return {"customers": [cid]}

        if event.event_type == "VEHICLE_BREAKDOWN":
            vid = int(event.payload["vehicle_id"])
            vehicle = self.instance.vehicles.get(vid)
            if vehicle is None:
                return {"vehicles": [], "customers": []}
            released: List[int] = []
            for cid in vehicle.route_suffix:
                customer = self.instance.customers.get(cid)
                if customer is not None and customer.status != SERVED:
                    customer.status = PENDING
                    customer.assigned_vehicle_id = None
                    released.append(cid)
            vehicle.available = False
            vehicle.status = BROKEN
            vehicle.route_suffix = []
            return {"vehicles": [vid], "customers": sorted(released)}

        if event.event_type == "CONGESTION_START":
            edges = event.payload.get("edges", [])
            multiplier = float(event.payload.get("travel_time_multiplier", 1.0))
            duration = float(event.payload.get("duration", 0.0))
            period_id = str(event.payload.get("period_id", event.event_id))
            self.travel.add_congestion(
                period_id=period_id,
                start=event.event_time,
                duration=duration,
                edges=edges,
                multiplier=multiplier,
            )
            self.network_version += 1
            return {
                "edges": [[int(a), int(b)] for a, b in edges],
                "congestion_period": period_id,
                "customers": [],
            }

        if event.event_type == "CONGESTION_END":
            self.travel.remove_congestion(str(event.payload.get("period_id", "")))
            self.network_version += 1
            return {"customers": []}
        return {"customers": []}

    # ------------------------------------------------------------------
    # 重规划
    # ------------------------------------------------------------------
    def _collect_remaining(self, now: float) -> List[int]:
        remaining: List[int] = []
        for cid, customer in self.instance.customers.items():
            if customer.status == SERVED:
                continue
            if customer.status == UNRELEASED and customer.release_time <= now + EPS:
                customer.status = PENDING
            if customer.status in (PENDING, ASSIGNED):
                remaining.append(cid)
        return sorted(remaining)

    def _old_assignment(self) -> Dict[int, Tuple[int, Optional[int]]]:
        """上一轮计划中的客户归属，用于计算路线调整率。"""
        mapping: Dict[int, Tuple[int, Optional[int]]] = {}
        for vid, route in self.last_plan.items():
            for pos, cid in enumerate(route):
                successor = route[pos + 1] if pos + 1 < len(route) else None
                mapping[cid] = (int(vid), successor)
        return mapping

    def _route_change_rate(
        self,
        old_map: Dict[int, Tuple[int, Optional[int]]],
        new_routes: Dict[int, List[int]],
        remaining: Sequence[int],
    ) -> float:
        if not old_map or not remaining:
            return 0.0
        new_map: Dict[int, Tuple[int, Optional[int]]] = {}
        for vid, route in new_routes.items():
            for pos, cid in enumerate(route):
                successor = route[pos + 1] if pos + 1 < len(route) else None
                new_map[cid] = (vid, successor)
        changed = 0
        considered = 0
        for cid in remaining:
            if cid not in old_map:
                continue
            considered += 1
            old_v, old_s = old_map[cid]
            new_v, new_s = new_map.get(cid, (None, None))
            if old_v != new_v or old_s != new_s:
                changed += 1
        return changed / considered if considered else 0.0

    def _apply_plan(self, solution: Optional[Solution]) -> Dict[int, List[int]]:
        """执行规划结果，并按累计剩余容量做最后一道硬约束截断。

        规划器自身已保证容量；截断只用于“无可行解”的兜底情形，
        保证仿真中任何车辆都不会物理超载，剩余订单留待下次重规划或记为未服务。
        """
        routes = solution.routes if solution is not None else {}
        applied: Dict[int, List[int]] = {}
        for vid, vehicle in self.instance.vehicles.items():
            vehicle.route_suffix = []
            if not vehicle.available or vehicle.status == BROKEN:
                continue
            cap = max(0.0, float(vehicle.capacity) - float(vehicle.cumulative_demand))
            seq: List[int] = []
            for cid in routes.get(vid, []):
                customer = self.instance.customers.get(cid)
                if customer is None:
                    continue
                if float(customer.demand) <= cap + 1e-9:
                    seq.append(cid)
                    cap -= float(customer.demand)
            if seq:
                applied[vid] = seq
                vehicle.route_suffix = list(seq)
                vehicle.status = ACTIVE
            elif vehicle.status == ACTIVE:
                vehicle.status = IDLE
        for customer in self.instance.customers.values():
            if customer.status == ASSIGNED:
                customer.status = PENDING
                customer.assigned_vehicle_id = None
        for vid, route in applied.items():
            for cid in route:
                customer = self.instance.customers.get(cid)
                if customer is not None and customer.status != SERVED:
                    customer.status = ASSIGNED
                    customer.assigned_vehicle_id = vid
        return applied

    def _run_planner(self, ctx: PlanningContext) -> AlgorithmResult:
        cfg = copy.deepcopy(self.planner_config)
        warm: List[Dict[int, List[int]]] = []
        if cfg.response_strategy == "local_repair":
            if self.last_plan:
                warm.append(self.last_plan)
        elif cfg.response_strategy == "warm_start":
            if self.last_plan:
                warm.append(self.last_plan)
            for sol in self.last_archive[:3]:
                warm.append(sol.routes)
        elif cfg.response_strategy == "mixed":
            if self.last_plan:
                warm.append(self.last_plan)
            for sol in self.last_archive[:6]:
                warm.append(sol.routes)
        # restart：不提供任何历史方案
        if self.algorithm == "greedy":
            return run_greedy(ctx, cfg, self.rng, warm)
        planner = get_planner(self.algorithm)
        return planner(ctx, cfg, self.rng, warm)

    def _replan(
        self,
        event_type: str,
        event_id: str,
        event_time: float,
        impact: Optional[Dict[str, object]] = None,
    ) -> None:
        impact = impact or {}
        now = max(
            float(event_time),
            max((v.current_time for v in self.instance.vehicles.values()), default=0.0),
        )
        self.clock = max(self.clock, now)
        self._release_due(now)
        # 所有车辆在事件时刻冻结：已完成的步锁定，未完成的行程从当前节点
        # 在 now 之后重新出发。未出车车辆也统一从 now 起可调度。
        for vehicle in self.instance.vehicles.values():
            if vehicle.status != BROKEN:
                vehicle.current_time = max(vehicle.current_time, now)
        old_map = self._old_assignment()
        remaining = self._collect_remaining(now)
        for vehicle in self.instance.vehicles.values():
            vehicle.route_suffix = []
        for cid in remaining:
            customer = self.instance.customers[cid]
            if customer.status != SERVED:
                customer.status = PENDING
                customer.assigned_vehicle_id = None

        ctx = PlanningContext.from_state(
            instance=self.instance,
            travel=self.travel,
            remaining=remaining,
            used_vehicle_ids=self.used_vehicle_ids,
            max_route_duration=self.config.max_route_duration,
            hard_time_windows=self.config.hard_time_windows,
            vehicles=self.instance.vehicles,
        )
        started = time.perf_counter()
        evals_before = self.evaluations_elapsed
        result = self._run_planner(ctx)
        wall = time.perf_counter() - started
        self.evaluations_elapsed += int(result.evaluations)

        feasible_candidates = [s for s in result.archive if s.feasible]
        chosen = select_compromise(feasible_candidates)
        if chosen is None:
            chosen = result.best_infeasible
        applied = self._apply_plan(chosen)
        plan_ok = bool(chosen is not None and chosen.feasible)
        if not plan_ok:
            self.had_infeasible_plan = True
        self.last_plan = {vid: list(r) for vid, r in applied.items()}
        self.last_archive = list(result.archive)
        self.archive_history.append(
            {"event_id": event_id, "solutions": list(result.archive)}
        )
        change_rate = self._route_change_rate(old_map, self.last_plan, remaining)
        applied_unserved = sum(
            1
            for cid in remaining
            if self.instance.customers[cid].status not in (SERVED, ASSIGNED)
        )
        available_vehicle_ids = [
            vid
            for vid, vehicle in self.instance.vehicles.items()
            if vehicle.available and vehicle.status != BROKEN
        ]
        vehicle_loads = {
            str(vid): float(vehicle.cumulative_demand / vehicle.capacity)
            if vehicle.capacity > 0
            else 0.0
            for vid, vehicle in self.instance.vehicles.items()
        }

        affected_customers = sorted(
            int(cid) for cid in impact.get("customers", [])  # type: ignore[union-attr]
        )
        if event_type != "INITIAL" and affected_customers:
            self.pending_recoveries.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "event_time": float(event_time),
                    "customers": affected_customers,
                    "evaluations_start": int(evals_before),
                }
            )
        self._update_recoveries(float(now))

        record = {
            "event_id": event_id,
            "event_type": event_type,
            "event_time": float(event_time),
            "planning_time": float(now),
            "wall_time_s": float(wall),
            "algorithm": self.algorithm,
            "evaluations": int(result.evaluations),
            "archive_size": len(result.archive),
            "feasible": plan_ok,
            "unserved": int(applied_unserved),
            "objectives": [float(x) for x in chosen.objectives] if chosen else None,
            "violation_score": float(chosen.violation_score) if chosen else None,
            "plan": {str(vid): list(route) for vid, route in applied.items()},
            "route_change_rate": float(change_rate),
            "remaining_customers": len(remaining),
            "affected_customers": affected_customers,
            "affected_vehicles": sorted(int(v) for v in impact.get("vehicles", [])),  # type: ignore[union-attr]
            "affected_edges": [list(e) for e in impact.get("edges", [])],  # type: ignore[union-attr]
            "network_version": int(self.network_version),
            "available_vehicles": len(available_vehicle_ids),
            "vehicle_loads": vehicle_loads,
            "response_strategy": self.planner_config.response_strategy,
        }
        self.replans.append(record)

    def _update_recoveries(self, now: float) -> None:
        """受影响订单首次全部恢复为已分配/已服务时，记录恢复时延与评价次数。"""
        still_pending: List[Dict[str, object]] = []
        for item in self.pending_recoveries:
            customers = [
                self.instance.customers[cid]
                for cid in item["customers"]  # type: ignore[union-attr]
                if cid in self.instance.customers
            ]
            recovered = bool(customers) and all(
                c.status in (ASSIGNED, SERVED) for c in customers
            )
            if recovered:
                self.recovery_records.append(
                    {
                        "event_id": item["event_id"],
                        "event_type": item["event_type"],
                        "event_time": item["event_time"],
                        "customers": list(item["customers"]),  # type: ignore[arg-type]
                        "recovered": True,
                        "recovery_time": float(now - float(item["event_time"])),
                        "recovery_evaluations": int(
                            self.evaluations_elapsed - int(item["evaluations_start"])
                        ),
                    }
                )
            else:
                still_pending.append(item)
        self.pending_recoveries = still_pending

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, object]:
        started = time.perf_counter()
        self._advance_to(0.0)
        self._release_due(0.0)
        # 时间为负/零的事件先处理
        events = sorted(self.scenario.events, key=lambda e: (e.event_time, e.event_id))
        idx = 0
        initial_impact: Dict[str, object] = {"customers": []}
        while idx < len(events) and events[idx].event_time <= EPS:
            impact = self._apply_event(events[idx])
            initial_impact["customers"] = list(initial_impact.get("customers", [])) + list(  # type: ignore[arg-type]
                impact.get("customers", [])
            )
            idx += 1
        self._replan("INITIAL", "initial", 0.0, initial_impact)

        for event in events[idx:]:
            self._advance_to(event.event_time)
            impact = self._apply_event(event)
            self._replan(event.event_type, event.event_id, event.event_time, impact)

        # 执行全部剩余路线；事件窗口结束不等于硬性截断路线。
        self._advance_to(float("inf"))
        # 未故障车辆返回仓库
        for vehicle in self.instance.vehicles.values():
            if vehicle.status == BROKEN or vehicle.current_node == self.instance.depot_id:
                continue
            departure = vehicle.current_time
            vehicle.cumulative_distance += self.travel.distance(
                vehicle.current_node, self.instance.depot_id
            )
            vehicle.current_time = departure + self.travel.travel_time(
                vehicle.current_node, self.instance.depot_id, departure
            )
            vehicle.current_node = self.instance.depot_id
            vehicle.status = FINISHED

        final_time = max(
            [v.current_time for v in self.instance.vehicles.values() if v.used]
            + [self.clock, 0.0]
        )
        # 最终仍未恢复的事件必须显式记录，不能静默当作成功。
        for item in self.pending_recoveries:
            self.recovery_records.append(
                {
                    "event_id": item["event_id"],
                    "event_type": item["event_type"],
                    "event_time": item["event_time"],
                    "customers": list(item["customers"]),  # type: ignore[arg-type]
                    "recovered": False,
                    "recovery_time": None,
                    "recovery_evaluations": None,
                }
            )
        self.pending_recoveries = []
        served = []
        unserved = []
        released = []
        total_lateness = 0.0
        total_early_wait = 0.0
        total_service_wait = 0.0
        for cid, customer in self.instance.customers.items():
            if customer.release_time <= final_time + EPS:
                released.append(cid)
            if customer.status == SERVED:
                served.append(cid)
                if customer.due_time is not None and customer.service_start_time is not None:
                    total_lateness += max(
                        0.0, float(customer.service_start_time) - float(customer.due_time)
                    )
                total_early_wait += max(
                    0.0,
                    float(customer.ready_time) - float(customer.arrival_time or customer.ready_time),
                )
                if customer.service_start_time is not None and customer.arrival_time is not None:
                    total_service_wait += max(
                        0.0, float(customer.service_start_time) - float(customer.arrival_time)
                    )
            elif customer.release_time <= final_time + EPS:
                customer.status = UNSERVED
                unserved.append(cid)

        used_vehicles = [
            v for v in self.instance.vehicles.values()
            if v.used or v.cumulative_demand > EPS
        ]
        # f1 与 evaluator 口径一致：已完成弧段 + 收工返回配送中心的距离。
        # 故障车辆不再返仓，其已执行距离仍保留。
        return_distance = sum(
            self.travel.distance(v.current_node, self.instance.depot_id)
            for v in used_vehicles
            if v.available
            and v.status != BROKEN
            and v.current_node != self.instance.depot_id
            and not v.route_suffix
        )
        f1 = sum(v.cumulative_distance for v in self.instance.vehicles.values()) + return_distance
        f2 = float(len(used_vehicles))
        f3 = sum(v.cumulative_response for v in used_vehicles)
        utils = [v.cumulative_demand / v.capacity for v in used_vehicles if v.capacity > 0]
        if utils:
            mean_u = sum(utils) / len(utils)
            f4 = sum((u - mean_u) ** 2 for u in utils) / len(utils)
        else:
            f4 = 0.0
        # 辅助口径（实验设计方案 docx）：所有可用车辆参与方差，供论文对照；
        # 主目标 f4 保持 md 的“已启用车辆利用率方差”。
        available_vehicles = [
            v for v in self.instance.vehicles.values()
            if v.available and v.status != BROKEN
        ]
        all_utils = [
            v.cumulative_demand / v.capacity for v in available_vehicles if v.capacity > 0
        ]
        all_loads = [v.cumulative_demand for v in available_vehicles]
        aux_util_var = _variance(all_utils)
        aux_load_var = _variance(all_loads)

        metrics = {
            "f1_total_distance": float(f1),
            "f2_used_vehicles": float(f2),
            "f3_total_response_time": float(f3),
            "f4_load_utilization_variance": float(f4),
            "released_orders": len(released),
            "served_orders": len(served),
            "unserved_orders": len(unserved),
            "service_rate": (len(served) / len(released)) if released else 1.0,
            "makespan": float(final_time),
            "feasible": len(unserved) == 0,
            "had_infeasible_plan": bool(self.had_infeasible_plan),
            "total_lateness": float(total_lateness),
            "total_early_wait": float(total_early_wait),
            "total_service_wait": float(total_service_wait),
            "load_utilization_variance_all_available": float(aux_util_var),
            "load_variance_all_available": float(aux_load_var),
            "replans": len(self.replans),
        }
        route_changes = [r["route_change_rate"] for r in self.replans if r["event_type"] != "INITIAL"]
        metrics["route_change_rate_mean"] = (
            float(sum(route_changes) / len(route_changes)) if route_changes else 0.0
        )
        replan_times = [r["wall_time_s"] for r in self.replans]
        metrics["replan_time_mean"] = float(sum(replan_times) / len(replan_times)) if replan_times else 0.0
        metrics["replan_time_max"] = float(max(replan_times)) if replan_times else 0.0
        metrics["evaluations_total"] = int(sum(r["evaluations"] for r in self.replans))

        recovered = [r for r in self.recovery_records if r["recovered"]]
        recovery_times = [float(r["recovery_time"]) for r in recovered]
        recovery_evals = [int(r["recovery_evaluations"]) for r in recovered]
        metrics["recovery_events"] = len(self.recovery_records)
        metrics["recovered_events"] = len(recovered)
        metrics["recovery_rate"] = (
            len(recovered) / len(self.recovery_records) if self.recovery_records else 1.0
        )
        metrics["recovery_time_mean"] = (
            float(sum(recovery_times) / len(recovery_times)) if recovery_times else 0.0
        )
        metrics["recovery_time_max"] = float(max(recovery_times)) if recovery_times else 0.0
        metrics["recovery_evaluations_mean"] = (
            float(sum(recovery_evals) / len(recovery_evals)) if recovery_evals else 0.0
        )
        metrics["unrecovered_events"] = len(self.recovery_records) - len(recovered)

        customers_snapshot = {
            str(cid): {
                "customer_id": cid,
                "x": c.x,
                "y": c.y,
                "demand": c.demand,
                "release_time": c.release_time,
                "status": c.status,
                "service_start_time": c.service_start_time,
                "arrival_time": c.arrival_time,
                "assigned_vehicle_id": c.assigned_vehicle_id,
            }
            for cid, c in self.instance.customers.items()
        }
        vehicles_snapshot = {
            str(vid): {
                "capacity": v.capacity,
                "used": v.used,
                "status": v.status,
                "current_node": v.current_node,
                "current_time": v.current_time,
                "cumulative_distance": v.cumulative_distance,
                "cumulative_demand": v.cumulative_demand,
                "served_customers": list(v.served_customers),
            }
            for vid, v in self.instance.vehicles.items()
        }

        return {
            "run": {
                "algorithm": self.algorithm,
                "instance_id": self.instance.instance_id,
                "scenario_id": self.scenario.scenario_id,
                "seed": self.seed,
            },
            "metrics": metrics,
            "replans": self.replans,
            "recovery": self.recovery_records,
            "final_plan": {str(vid): list(route) for vid, route in self.last_plan.items()},
            "archive": [
                {
                    "objectives": [float(x) for x in s.objectives],
                    "feasible": s.feasible,
                    "routes": {str(vid): list(r) for vid, r in s.routes.items()},
                }
                for s in self.last_archive
            ],
            # 每次重规划的完整可行档案：用于按事件计算 HV/IGD，而不是只看最后一次。
            "archive_history": [
                {
                    "event_id": item["event_id"],
                    "objectives": [
                        [float(x) for x in s.objectives] for s in item["solutions"]
                    ],
                }
                for item in self.archive_history
            ],
            "customers": customers_snapshot,
            "vehicles": vehicles_snapshot,
            "congestion_periods": self.travel.congestion_snapshot(),
            "wall_time_s": float(time.perf_counter() - started),
            "sim_config": {
                "horizon": self.config.horizon,
                "speed": self.config.speed,
                "max_route_duration": self.config.max_route_duration,
                "hard_time_windows": self.config.hard_time_windows,
                "selection_rule": self.config.selection_rule,
            },
            "planner_config": {
                "population_size": self.planner_config.population_size,
                "max_evaluations": self.planner_config.max_evaluations,
                "archive_size": self.planner_config.archive_size,
                "crossover_rate": self.planner_config.crossover_rate,
                "mutation_rate": self.planner_config.mutation_rate,
                "response_strategy": self.planner_config.response_strategy,
                "pheromone_evaporation": self.planner_config.pheromone_evaporation,
            },
        }

"""合成实例与动态事件日程生成。

说明：动态事件是本项目合成的可复现测试场景，不是 Solomon/CVRPLIB 官方动态基准。
事件随机数流与算法随机数流严格分离：事件文件先生成并保存，三类算法读取同一文件。
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .instance_loader import save_instance, save_scenario
from .models import Customer, DynamicEvent, PENDING, Scenario, UNRELEASED, VRPInstance, Vehicle


GENERATOR_VERSION = "1.1"
SCENARIO_SEED_STRIDE = 1009


def estimate_operation_horizon(instance: VRPInstance, speed: float = 1.0) -> float:
    """用确定性贪心方案估计一个规划周期。

    事件时间以该周期为比例，避免在车队早已收工后才释放订单或制造故障。
    这只是场景时间尺度，不是最优完工时间。
    """
    # 延迟导入，避免 scenario_generator 与 objectives 的循环依赖。
    from .context import PlanningContext
    from .objectives import evaluate_plan
    from .solution import greedy_solution
    from .travel_time import TravelNetwork

    travel = TravelNetwork(instance, speed=speed)
    ctx = PlanningContext.from_state(
        instance=instance,
        travel=travel,
        remaining=list(instance.customers.keys()),
        used_vehicle_ids=set(),
    )
    routes = greedy_solution(ctx, None)
    evaluation = evaluate_plan(routes, ctx)
    finish_times = [
        float(detail["finish_time"])
        for detail in evaluation.vehicle_details.values()
        if detail.get("finish_time") is not None
    ]
    makespan = max(finish_times) if finish_times else 0.0
    return max(60.0, makespan * 1.25)


def generate_toy_instance(
    instance_id: str,
    num_customers: int,
    num_vehicles: int,
    capacity: float,
    seed: int,
    grid: float = 100.0,
    depot: Tuple[float, float] = (50.0, 50.0),
    source: str = "synthetic-toy",
) -> VRPInstance:
    rng = np.random.default_rng(seed)
    customers: Dict[int, Customer] = {}
    for cid in range(1, num_customers + 1):
        customers[cid] = Customer(
            customer_id=cid,
            x=round(float(rng.uniform(5, grid - 5)), 2),
            y=round(float(rng.uniform(5, grid - 5)), 2),
            demand=float(rng.integers(5, 21)),
            release_time=0.0,
            service_time=float(rng.integers(2, 7)),
            ready_time=0.0,
            due_time=None,
            status=PENDING,
        )
    vehicles = {
        vid: Vehicle(
            vehicle_id=vid,
            capacity=float(capacity),
            start_time=0.0,
            current_node=0,
            current_time=0.0,
        )
        for vid in range(num_vehicles)
    }
    return VRPInstance(
        instance_id=instance_id,
        depot_id=0,
        depot_x=depot[0],
        depot_y=depot[1],
        vehicles=vehicles,
        customers=customers,
        source=source,
        metadata={"seed": seed, "generator_version": GENERATOR_VERSION},
    )


def generate_toy_instances(output_dir: str, specs: Sequence[Dict[str, object]]) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for spec in specs:
        instance = generate_toy_instance(**spec)  # type: ignore[arg-type]
        path = os.path.join(output_dir, "%s.json" % instance.instance_id)
        save_instance(instance, path)
        paths.append(path)
    return paths


def _edge_list(instance: VRPInstance) -> List[Tuple[int, int]]:
    nodes = [instance.depot_id] + sorted(instance.customers.keys())
    edges: List[Tuple[int, int]] = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            edges.append((min(a, b), max(a, b)))
    return edges


def generate_scenario(
    instance: VRPInstance,
    scenario_id: str,
    seed: int,
    horizon: Optional[float] = None,
    new_order_ratio: float = 0.0,
    breakdown_count: int = 0,
    breakdown_ratio: float = 0.0,
    congestion_edge_ratio: float = 0.0,
    congestion_multiplier: float = 1.5,
    congestion_duration_ratio: float = 0.25,
    simultaneous: bool = False,
    speed: float = 1.0,
) -> Scenario:
    """按强度参数生成事件日程。

    - horizon=None 时按实例的贪心方案完工时间自动估计，保证事件发生在运营期内。
    - new_order_ratio: 新增订单数 / 初始客户数。
    - breakdown_count: 故障车辆数。
    - congestion_edge_ratio: 受影响无向边比例。
    - 所有事件时间均相对规划周期 horizon。
    """
    if horizon is None or float(horizon) <= 0:
        horizon = estimate_operation_horizon(instance, speed=speed)
    horizon = float(horizon)
    rng = np.random.default_rng(seed + 7919)
    events: List[DynamicEvent] = []
    counter = 1

    def add_event(t: float, etype: str, payload: Dict[str, object]) -> None:
        nonlocal counter
        events.append(
            DynamicEvent(
                event_id="e%03d" % counter,
                event_type=etype,
                event_time=round(float(t), 2),
                payload=payload,
            )
        )
        counter += 1

    num_new = int(round(new_order_ratio * len(instance.customers)))
    if new_order_ratio > 0 and num_new == 0:
        num_new = 1
    xs = [instance.depot_x] + [c.x for c in instance.customers.values()]
    ys = [instance.depot_y] + [c.y for c in instance.customers.values()]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_pad = 0.05 * max(1.0, x_hi - x_lo)
    y_pad = 0.05 * max(1.0, y_hi - y_lo)
    if num_new > 0:
        base_times = np.linspace(0.10 * horizon, 0.55 * horizon, num=max(1, num_new))
        for idx in range(num_new):
            t = float(base_times[idx] + rng.uniform(-0.02, 0.02) * horizon)
            t = max(1.0, min(t, 0.80 * horizon))
            cid = 1001 + idx
            add_event(
                t,
                "NEW_ORDER",
                {
                    "customer": {
                        "customer_id": cid,
                        "x": round(float(rng.uniform(x_lo - x_pad, x_hi + x_pad)), 2),
                        "y": round(float(rng.uniform(y_lo - y_pad, y_hi + y_pad)), 2),
                        "demand": float(rng.integers(5, 21)),
                        "service_time": float(rng.integers(2, 7)),
                        "release_time": round(t, 2),
                    }
                },
            )

    vehicle_ids = sorted(instance.vehicles.keys())
    if breakdown_count <= 0 and breakdown_ratio > 0:
        breakdown_count = max(1, int(round(breakdown_ratio * len(vehicle_ids))))
    if breakdown_count > 0:
        chosen = list(rng.choice(vehicle_ids, size=min(breakdown_count, len(vehicle_ids)), replace=False))
        fractions = np.linspace(0.35, 0.55, num=len(chosen)) if not simultaneous else [0.40] * len(chosen)
        for vid, frac in zip(chosen, fractions):
            add_event(
                float(frac * horizon + rng.uniform(-0.02, 0.02) * horizon),
                "VEHICLE_BREAKDOWN",
                {"vehicle_id": int(vid), "repairable": False},
            )

    if congestion_edge_ratio > 0:
        edges = _edge_list(instance)
        num_edges = max(1, int(round(congestion_edge_ratio * len(edges))))
        selected = [edges[i] for i in rng.choice(len(edges), size=num_edges, replace=False)]
        periods = 1 if simultaneous else 2
        for p in range(periods):
            start = (0.15 + 0.30 * p) * horizon
            add_event(
                start,
                "CONGESTION_START",
                {
                    "period_id": "%s_cong_%d" % (scenario_id, p + 1),
                    "edges": [[int(a), int(b)] for a, b in selected],
                    "travel_time_multiplier": float(congestion_multiplier),
                    "duration": float(congestion_duration_ratio * horizon),
                },
            )

    events.sort(key=lambda e: (e.event_time, e.event_id))
    for idx, e in enumerate(events, start=1):
        e.event_id = "e%03d" % idx

    return Scenario(
        scenario_id=scenario_id,
        seed=int(seed),
        events=events,
        intensity={
            "new_order_ratio": float(new_order_ratio),
            "breakdown_count": int(breakdown_count),
            "congestion_edge_ratio": float(congestion_edge_ratio),
            "congestion_multiplier": float(congestion_multiplier),
        },
        metadata={
            "instance_id": instance.instance_id,
            "horizon": horizon,
            "generator_version": GENERATOR_VERSION,
            "note": "合成动态事件，非官方动态基准",
        },
    )


def scenario_seed(base_seed: int, run_seed: int, spec_index: int) -> int:
    """事件随机流与算法随机流分离：事件种子只由 (base, run seed, 场景序号) 决定。"""
    return int(base_seed + int(run_seed) * SCENARIO_SEED_STRIDE + int(spec_index) * 17)


def generate_scenario_suite(
    instance: VRPInstance,
    output_dir: str,
    specs: Sequence[Dict[str, object]],
    base_seed: int = 2026,
    horizon: Optional[float] = None,
    seeds: Sequence[int] = (0,),
    speed: float = 1.0,
) -> List[str]:
    """为每个（场景, 运行种子）生成独立事件文件。

    同一 (实例, 场景, 种子) 下三种算法读取完全相同的文件，用于配对比较。
    """
    os.makedirs(output_dir, exist_ok=True)
    if horizon is None or float(horizon) <= 0:
        horizon = estimate_operation_horizon(instance, speed=speed)
    horizon = float(horizon)
    paths: List[str] = []
    for idx, spec in enumerate(specs):
        scenario_id = str(spec["id"])
        for run_seed in seeds:
            event_seed = scenario_seed(base_seed, int(run_seed), idx)
            scenario = generate_scenario(
                instance=instance,
                scenario_id=scenario_id,
                seed=event_seed,
                horizon=horizon,
                new_order_ratio=float(spec.get("new_order_ratio", 0.0)),
                breakdown_count=int(spec.get("breakdown_count", 0)),
                breakdown_ratio=float(spec.get("breakdown_ratio", 0.0)),
                congestion_edge_ratio=float(spec.get("congestion_edge_ratio", 0.0)),
                congestion_multiplier=float(spec.get("congestion_multiplier", 1.0)),
                congestion_duration_ratio=float(spec.get("congestion_duration_ratio", 0.25)),
                simultaneous=bool(spec.get("simultaneous", False)),
                speed=speed,
            )
            path = os.path.join(
                output_dir,
                "%s__%s__s%d.json" % (instance.instance_id, scenario_id, int(run_seed)),
            )
            save_scenario(scenario, path)
            paths.append(path)
    return paths

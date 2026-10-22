"""唯一的四目标评估器。

目标顺序固定（全部最小化）：
f1 总行驶距离；f2 规划周期内启用车辆数；f3 订单响应时长总和；f4 容量利用率方差。

已执行距离/响应时长来自仿真状态常量，候选方案只影响剩余任务。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .constraints import check_routes
from .context import PlanningContext
from .models import Evaluation


EPS = 1e-9


def evaluate_plan(routes: Dict[int, List[int]], ctx: PlanningContext) -> Evaluation:
    check = check_routes(routes, ctx)
    violations = list(check.violations)
    score = check.violation_score

    planned_distance = 0.0
    planned_response = 0.0
    planned_lateness = 0.0
    planned_early = 0.0
    vehicle_details: Dict[int, Dict[str, object]] = {}
    used_ids = set(ctx.used_vehicle_ids)
    planned_loads: Dict[int, float] = {}

    for vid, route in routes.items():
        view = ctx.vehicles.get(vid)
        if view is None:
            continue
        if route and view.available:
            used_ids.add(vid)
        node = view.start_node
        t = view.start_time
        distance = 0.0
        response = 0.0
        load = 0.0
        starts: Dict[int, float] = {}
        if route:
            for cid in route:
                customer = ctx.instance.customers.get(cid)
                if customer is None or cid not in check.seen:
                    continue
                tau = ctx.travel.travel_time(node, cid, t)
                arrival = t + tau
                start = max(arrival, float(customer.release_time), float(customer.ready_time))
                response += start - float(customer.release_time)
                if customer.due_time is not None:
                    late = max(0.0, start - float(customer.due_time))
                    planned_lateness += late
                    if late > EPS and ctx.hard_time_windows:
                        violations.append(
                            "客户 %s 违反硬时间窗: %.2f > %.2f"
                            % (cid, start, customer.due_time)
                        )
                        score += late
                planned_early += max(0.0, float(customer.ready_time) - arrival)
                distance += ctx.travel.distance(node, cid)
                starts[cid] = start
                t = start + float(customer.service_time)
                load += float(customer.demand)
                node = cid
            back = ctx.travel.distance(node, ctx.instance.depot_id)
            distance += back
            t += ctx.travel.travel_time(node, ctx.instance.depot_id, t)
            if ctx.max_route_duration is not None:
                duration = t - view.start_time
                if duration > float(ctx.max_route_duration) + EPS:
                    violations.append(
                        "车辆 %s 路线时长 %.1f 超过上限 %.1f"
                        % (vid, duration, ctx.max_route_duration)
                    )
                    score += duration - float(ctx.max_route_duration)
        elif view.start_node != ctx.instance.depot_id and view.available:
            # 已出车但当前无任务的车辆最终仍需返回仓库；该距离对所有方案是常量。
            distance += ctx.travel.distance(view.start_node, ctx.instance.depot_id)

        planned_distance += distance
        planned_response += response
        planned_loads[vid] = load
        vehicle_details[vid] = {
            "load": load,
            "distance": distance,
            "finish_time": t,
            "response": response,
            "customer_start_times": starts,
        }

    base_distance = sum(v.cumulative_distance for v in ctx.vehicles.values())
    base_response = sum(v.cumulative_response for v in ctx.vehicles.values())
    f1 = base_distance + planned_distance
    f2 = float(len(used_ids))
    f3 = base_response + planned_response

    utilizations: List[float] = []
    for vid in sorted(used_ids):
        view = ctx.vehicles.get(vid)
        if view is None:
            continue
        total_load = view.cumulative_demand + planned_loads.get(vid, 0.0)
        if view.capacity > 0:
            utilizations.append(total_load / view.capacity)
    if utilizations:
        mean_u = sum(utilizations) / len(utilizations)
        f4 = sum((u - mean_u) ** 2 for u in utilizations) / len(utilizations)
    else:
        f4 = 0.0

    feasible = not violations
    return Evaluation(
        objectives=(float(f1), float(f2), float(f3), float(f4)),
        feasible=feasible,
        violations=violations,
        unserved=list(check.unserved),
        violation_score=float(score),
        vehicle_details=vehicle_details,
        total_lateness=float(planned_lateness),
        total_early_wait=float(planned_early),
        served_count=len(check.seen),
    )


def objective_vector(evaluation: Evaluation) -> Tuple[float, ...]:
    return tuple(float(x) for x in evaluation.objectives)

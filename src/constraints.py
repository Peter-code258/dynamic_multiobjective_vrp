"""独立约束检查器。

只做结构与容量检查；时间递推、时间窗与路线时长由 objectives 统一计算。
所有算法和最终仿真共用此检查器，避免不同实现采用不同可行性口径。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .context import PlanningContext


@dataclass
class CheckResult:
    violations: List[str] = field(default_factory=list)
    unserved: List[int] = field(default_factory=list)
    loads: Dict[int, float] = field(default_factory=dict)
    seen: Set[int] = field(default_factory=set)
    violation_score: float = 0.0

    @property
    def feasible(self) -> bool:
        return not self.violations


def check_routes(routes: Dict[int, List[int]], ctx: PlanningContext) -> CheckResult:
    result = CheckResult()
    remaining = set(ctx.remaining)
    n = max(1, len(remaining))
    score = 0.0
    seen: Set[int] = set()

    for vid, route in routes.items():
        vehicle = ctx.vehicles.get(vid)
        if vehicle is None:
            result.violations.append("未知车辆 %s" % vid)
            score += 1000.0
            continue
        if not vehicle.available:
            result.violations.append("车辆 %s 不可用但仍被分配任务" % vid)
            score += 1000.0
        load = 0.0
        for cid in route:
            if cid not in remaining:
                result.violations.append("客户 %s 不在待规划集合或已被服务" % cid)
                score += 1000.0
                continue
            if cid in seen:
                result.violations.append("客户 %s 被重复服务" % cid)
                score += 100.0
                continue
            seen.add(cid)
            customer = ctx.instance.customers.get(cid)
            if customer is None:
                result.violations.append("未知客户 %s" % cid)
                score += 1000.0
                continue
            load += customer.demand
        result.loads[vid] = load
        total_load = vehicle.cumulative_demand + load
        excess = total_load - vehicle.capacity
        if excess > 1e-9:
            result.violations.append(
                "车辆 %s 累计超载: %.2f > %.2f（已服务 %.2f + 本段 %.2f）"
                % (vid, total_load, vehicle.capacity, vehicle.cumulative_demand, load)
            )
            score += 10.0 * excess

    missing = sorted(remaining - seen)
    if missing:
        result.unserved = missing
        result.violations.append("未服务客户 %d 个: %s" % (len(missing), missing[:20]))
        score += (n + 1) * 100.0 * len(missing)

    result.seen = seen
    result.violation_score = score
    return result

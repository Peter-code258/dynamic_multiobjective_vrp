"""动态贪心插入基线（工程参照，不参与 Pareto 前沿质量排名）。"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..context import PlanningContext, remaining_capacity
from ..pareto import ParetoArchive, best_infeasible
from ..solution import _best_insertion, _new_route_cost, repair_routes
from ..algorithms.base import AlgorithmResult, EvaluationBudget, PlannerConfig, make_result


def run_greedy(
    ctx: PlanningContext,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    warm_routes: Optional[Sequence[Dict[int, List[int]]]] = None,
) -> AlgorithmResult:
    started = time.perf_counter()
    budget = EvaluationBudget(cfg.max_evaluations)
    routes: Dict[int, List[int]] = {}
    unused = list(ctx.available_vehicle_ids)
    order = sorted(
        ctx.remaining,
        key=lambda c: (
            ctx.instance.customers[c].release_time,
            -ctx.instance.customers[c].demand,
            c,
        ),
    )
    for cid in order:
        insert = _best_insertion(routes, cid, ctx, None) if routes else None
        if insert is not None:
            _, vid, pos = insert
            routes[vid].insert(pos, cid)
            continue
        if unused:
            fit = [
                v for v in unused
                if ctx.instance.customers[cid].demand
                <= remaining_capacity(ctx.vehicles[v]) + 1e-9
            ]
            if fit:
                vid = min(fit, key=lambda v: _new_route_cost(v, cid, ctx, None))
                routes[vid] = [cid]
                unused.remove(vid)
                continue
        # 无可用容量：保留为未服务，由评估器记录。
    routes = repair_routes(routes, ctx, None)
    sol = budget.evaluate(routes, ctx)
    archive = ParetoArchive(max_size=cfg.archive_size)
    if sol.feasible:
        archive.add(sol)
    return make_result(
        "greedy",
        archive.solutions,
        best_infeasible([sol]),
        budget.evaluations,
        started,
        cfg,
        {"archive_size": len(archive)},
    )

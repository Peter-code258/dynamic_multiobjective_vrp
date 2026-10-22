"""离散 MOPSO。

不把连续实数位置套到客户编号上；位置是车辆路线集合，速度是一组离散变换
（插入、交换、反转、跨路线迁移、拆分/合并）。每次更新按惯性扰动、
个体历史最优吸引和外部档案领导者吸引依次施加离散变换。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..context import PlanningContext
from ..models import Solution
from ..pareto import ParetoArchive, best_infeasible, constrained_dominates
from ..solution import clone_routes, crossover_routes, mutate_routes, repair_routes
from .base import (
    AlgorithmResult,
    BudgetExhausted,
    EvaluationBudget,
    PlannerConfig,
    initial_route_seeds,
    make_result,
)


def _pbest_update(current: Solution, candidate: Solution, rng: np.random.Generator) -> Solution:
    if constrained_dominates(candidate, current):
        return candidate
    if constrained_dominates(current, candidate):
        return current
    # 互不支配时保留一个随机样本，维持个体多样性。
    return candidate if rng.random() < 0.5 else current


def run_mopso(
    ctx: PlanningContext,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    warm_routes: Optional[Sequence[Dict[int, List[int]]]] = None,
) -> AlgorithmResult:
    started = time.perf_counter()
    budget = EvaluationBudget(cfg.max_evaluations)
    archive = ParetoArchive(max_size=cfg.archive_size, feasible_only=True)
    all_seen: List[Solution] = []

    pop_size = max(4, int(cfg.population_size))
    seeds = initial_route_seeds(ctx, cfg, rng, warm_routes or [], pop_size)
    particles: List[Solution] = []
    pbest: List[Solution] = []
    for routes in seeds:
        try:
            sol = budget.evaluate(routes, ctx)
        except BudgetExhausted:
            break
        particles.append(sol)
        pbest.append(sol.copy())
        all_seen.append(sol)
        if sol.feasible:
            archive.add(sol)
    if not particles:
        sol = budget.evaluate(seeds[0], ctx)
        particles.append(sol)
        pbest.append(sol.copy())

    c1 = float(cfg.extra.get("c1", 0.6))
    c2 = float(cfg.extra.get("c2", 0.6))
    inertia_strength = int(cfg.extra.get("inertia_strength", 2))
    iterations = 0

    while budget.remaining > 0:
        iterations += 1
        for idx in range(len(particles)):
            if budget.remaining <= 0:
                break
            # 惯性：离散扰动（速度的随机分量）
            pos = mutate_routes(
                clone_routes(particles[idx].routes),
                ctx,
                rng,
                strength=max(1, inertia_strength),
            )
            # 个体最优吸引：路线级交叉，等价于一次离散位置更新
            if rng.random() < c1:
                pos = crossover_routes(pos, pbest[idx].routes, ctx, rng)
            # 档案领导者吸引：从稀疏区域抽取领导者
            leader = archive.sample_leader(rng)
            if leader is not None and rng.random() < c2:
                pos = crossover_routes(pos, leader.routes, ctx, rng)
            pos = repair_routes(pos, ctx, rng)
            try:
                candidate = budget.evaluate(pos, ctx)
            except BudgetExhausted:
                break
            all_seen.append(candidate)
            particles[idx] = candidate
            pbest[idx] = _pbest_update(pbest[idx], candidate.copy(), rng)
            if candidate.feasible:
                archive.add(candidate)

    best_inf = best_infeasible(all_seen)
    return make_result(
        "mopso",
        archive.solutions,
        best_inf,
        budget.evaluations,
        started,
        cfg,
        {"iterations": iterations, "archive_size": len(archive)},
    )

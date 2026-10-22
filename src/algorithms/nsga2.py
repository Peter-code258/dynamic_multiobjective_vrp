"""NSGA-II：快速非支配排序、拥挤距离、精英保留与路线级遗传算子。"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..context import PlanningContext
from ..models import Solution
from ..pareto import ParetoArchive, best_infeasible, constrained_rank_and_crowding
from ..solution import clone_routes, crossover_routes, mutate_routes
from .base import (
    AlgorithmResult,
    BudgetExhausted,
    EvaluationBudget,
    PlannerConfig,
    initial_route_seeds,
    make_result,
)


def _tournament(
    population: Sequence[Solution],
    ranks: Sequence[int],
    crowd: Sequence[float],
    rng: np.random.Generator,
) -> Solution:
    n = len(population)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n))
    if ranks[i] < ranks[j]:
        return population[i]
    if ranks[j] < ranks[i]:
        return population[j]
    return population[i] if crowd[i] >= crowd[j] else population[j]


def _environmental_select(
    candidates: Sequence[Solution], size: int
) -> List[Solution]:
    ranks, crowd = constrained_rank_and_crowding(candidates)
    order = sorted(range(len(candidates)), key=lambda i: (ranks[i], -crowd[i], i))
    return [candidates[i] for i in order[:size]]


def run_nsga2(
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
    population: List[Solution] = []
    for routes in seeds:
        try:
            sol = budget.evaluate(routes, ctx)
        except BudgetExhausted:
            break
        population.append(sol)
        all_seen.append(sol)
        if sol.feasible:
            archive.add(sol)
    if not population:
        # 预算过小：至少返回一条可解释的构造解。
        sol = budget.evaluate(seeds[0], ctx)
        population.append(sol)
        all_seen.append(sol)
        if sol.feasible:
            archive.add(sol)

    generation = 0
    while budget.remaining > 0:
        generation += 1
        ranks, crowd = constrained_rank_and_crowding(population)
        offspring: List[Solution] = []
        n_offspring = min(pop_size, budget.remaining)
        for _ in range(n_offspring):
            p1 = _tournament(population, ranks, crowd, rng)
            p2 = _tournament(population, ranks, crowd, rng)
            if rng.random() < cfg.crossover_rate:
                child_routes = crossover_routes(p1.routes, p2.routes, ctx, rng)
            else:
                child_routes = clone_routes(p1.routes)
            if rng.random() < cfg.mutation_rate:
                child_routes = mutate_routes(child_routes, ctx, rng, strength=1)
            try:
                sol = budget.evaluate(child_routes, ctx)
            except BudgetExhausted:
                break
            offspring.append(sol)
            all_seen.append(sol)
            if sol.feasible:
                archive.add(sol)
        if not offspring:
            break
        population = _environmental_select(list(population) + offspring, pop_size)

    # 返回已发现的最优非支配可行解；若没有任何可行解，返回最小违反度解。
    best_inf = best_infeasible(all_seen)
    return make_result(
        "nsga2",
        archive.solutions,
        best_inf,
        budget.evaluations,
        started,
        cfg,
        {"generations": generation, "archive_size": len(archive)},
    )

"""算法公共接口、评价预算与初始种群构造。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..context import PlanningContext
from ..models import Evaluation, Solution
from ..objectives import evaluate_plan
from ..solution import greedy_solution, mutate_routes, random_solution, repair_routes


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class PlannerConfig:
    population_size: int = 50
    max_evaluations: int = 5000
    archive_size: int = 100
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    response_strategy: str = "warm_start"
    pheromone_evaporation: float = 0.2
    extra: Dict[str, float] = field(default_factory=dict)


class EvaluationBudget:
    """所有算法共享的候选解评价计数器；额外验证也计入预算。"""

    def __init__(self, max_evaluations: int) -> None:
        self.max_evaluations = int(max_evaluations)
        self.evaluations = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_evaluations - self.evaluations)

    def evaluate(self, routes: Dict[int, List[int]], ctx: PlanningContext) -> Solution:
        if self.evaluations >= self.max_evaluations:
            raise BudgetExhausted("评价预算已用尽")
        self.evaluations += 1
        ev: Evaluation = evaluate_plan(routes, ctx)
        return Solution(
            routes={vid: list(r) for vid, r in routes.items() if r},
            objectives=tuple(ev.objectives),
            feasible=ev.feasible,
            violation_score=ev.violation_score,
            violations=list(ev.violations),
            evaluation=ev,
        )


@dataclass
class AlgorithmResult:
    algorithm: str
    archive: List[Solution]
    best_infeasible: Optional[Solution]
    evaluations: int
    wall_time: float
    response_strategy: str
    metadata: Dict[str, object] = field(default_factory=dict)


def initial_route_seeds(
    ctx: PlanningContext,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    warm_routes: Sequence[Dict[int, List[int]]],
    count: int,
) -> List[Dict[int, List[int]]]:
    """混合初始解：按响应策略注入历史方案，再补足随机化最近邻与随机排列。

    - restart：完全不使用历史方案；
    - local_repair：只把上一条执行方案修复后作为种子；
    - warm_start：历史方案修复后注入；
    - mixed：历史方案及其变异副本同时注入，平衡开发与探索。
    """
    seeds: List[Dict[int, List[int]]] = []
    strategy = cfg.response_strategy
    if strategy == "mixed":
        for routes in warm_routes:
            if len(seeds) >= count:
                break
            repaired = repair_routes(routes, ctx, rng)
            seeds.append(repaired)
            if len(seeds) < count:
                seeds.append(mutate_routes(repaired, ctx, rng, strength=2))
    elif strategy != "restart":
        for routes in warm_routes:
            if len(seeds) >= count:
                break
            seeds.append(repair_routes(routes, ctx, rng))
    while len(seeds) < count:
        if len(seeds) % 3 == 0:
            seeds.append(greedy_solution(ctx, rng))
        elif len(seeds) % 3 == 1:
            seeds.append(random_solution(ctx, rng))
        else:
            seeds.append(greedy_solution(ctx, rng))
    return seeds[:count]


def make_result(
    algorithm: str,
    archive: Sequence[Solution],
    best_inf: Optional[Solution],
    evaluations: int,
    started: float,
    cfg: PlannerConfig,
    metadata: Optional[Dict[str, object]] = None,
) -> AlgorithmResult:
    return AlgorithmResult(
        algorithm=algorithm,
        archive=list(archive),
        best_infeasible=best_inf,
        evaluations=int(evaluations),
        wall_time=float(time.perf_counter() - started),
        response_strategy=cfg.response_strategy,
        metadata=dict(metadata or {}),
    )

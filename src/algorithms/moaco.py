"""Pareto 档案型 MOACO（本项目的明确变体）。

定义：四组目标信息素矩阵 + 随机 Dirichlet 目标权重 + 非支配外部档案
+ 按目标覆盖与稀疏度轮换的精英信息素更新。该定义是工程选择，
不代表 MOACO 存在唯一标准版本。
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..context import PlanningContext
from ..models import Solution
from ..pareto import ParetoArchive, best_infeasible
from ..solution import repair_routes
from .base import (
    AlgorithmResult,
    BudgetExhausted,
    EvaluationBudget,
    PlannerConfig,
    initial_route_seeds,
    make_result,
)


class PheromoneSet:
    def __init__(self, objectives: int = 4, tau0: float = 1.0) -> None:
        self.objectives = objectives
        self.tau0 = float(tau0)
        self.tau_min = 0.05
        self.tau_max = 10.0
        self.matrices: List[Dict[Tuple[int, int], float]] = [dict() for _ in range(objectives)]

    def get(self, m: int, i: int, j: int) -> float:
        return self.matrices[m].get((int(i), int(j)), self.tau0)

    def add(self, m: int, i: int, j: int, value: float) -> None:
        key = (int(i), int(j))
        current = self.matrices[m].get(key, self.tau0)
        self.matrices[m][key] = min(self.tau_max, max(self.tau_min, current + value))

    def evaporate(self, rho: float) -> None:
        rho = min(0.999, max(0.0, float(rho)))
        for matrix in self.matrices:
            for key in list(matrix.keys()):
                matrix[key] = max(self.tau_min, (1.0 - rho) * matrix[key])

    def deposit_solution(
        self,
        solution: Solution,
        ctx: PlanningContext,
        weight: float,
        objectives: Optional[Sequence[int]] = None,
    ) -> None:
        """把一条解沉积到指定目标的信息素矩阵（默认四组同时沉积）。

        只让某个目标上的精英解更新该目标矩阵，四组信息素才会形成差异；
        若四组矩阵始终同步更新，蚂蚁会迅速收敛到同一条路线，档案塌缩为单点。
        """
        targets = list(range(self.objectives)) if objectives is None else list(objectives)
        for vid, route in solution.routes.items():
            if not route:
                continue
            view = ctx.vehicles.get(vid)
            if view is None:
                continue
            prev = view.start_node
            for cid in route:
                for m in targets:
                    self.add(m, prev, cid, weight)
                prev = cid
            for m in targets:
                self.add(m, prev, ctx.instance.depot_id, weight)


def _heuristic_values(
    ctx: PlanningContext,
    node: int,
    t: float,
    load: float,
    cap: float,
    cid: int,
    target_util: float,
) -> List[float]:
    customer = ctx.instance.customers[cid]
    d = ctx.travel.distance(node, cid)
    arrival = t + ctx.travel.travel_time(node, cid, t)
    start = max(arrival, float(customer.release_time), float(customer.ready_time))
    response = max(0.0, start - float(customer.release_time))
    util_after = (load + float(customer.demand)) / cap if cap > 0 else 1.0
    eta1 = 1.0 / (d + 1.0)
    eta2 = min(1.0, util_after) + 0.05
    eta3 = 1.0 / (1.0 + response)
    eta4 = 1.0 / (1.0 + abs(util_after - target_util))
    return [eta1, eta2, eta3, eta4]


def _construct_ant(
    ctx: PlanningContext,
    rng: np.random.Generator,
    pheromones: PheromoneSet,
    weights: np.ndarray,
    alpha: float,
    beta: float,
    stop_probability: float = 0.25,
) -> Dict[int, List[int]]:
    unassigned: Set[int] = set(ctx.remaining)
    vehicles = list(ctx.available_vehicle_ids)
    rng.shuffle(vehicles)
    routes: Dict[int, List[int]] = {}
    total_demand = sum(ctx.instance.customers[c].demand for c in unassigned)
    target_util = min(
        1.0,
        total_demand / max(1e-9, sum(ctx.vehicles[v].capacity for v in vehicles)),
    )
    # 每只蚂蚁随机决定启用车队规模，并允许中途关闭当前路线。
    # 否则容量足够时第一辆车会吞下全部客户，f2 永远等于 1，档案无法体现
    # “少用车 vs 低响应时长”的折中。
    target_vehicles = int(rng.integers(1, len(vehicles) + 1)) if vehicles else 0
    active = vehicles[:target_vehicles]
    stop_prob = float(stop_probability)

    for idx, vid in enumerate(active):
        if not unassigned:
            break
        view = ctx.vehicles[vid]
        node = view.start_node
        current_time = view.start_time
        load = 0.0
        seq: List[int] = []
        is_last = idx == len(active) - 1
        while unassigned:
            candidates = [
                c for c in unassigned
                if load + ctx.instance.customers[c].demand <= view.capacity + 1e-9
            ]
            if not candidates:
                break
            scores: List[float] = []
            for cid in candidates:
                etas = _heuristic_values(
                    ctx, node, current_time, load, view.capacity, cid, target_util
                )
                log_score = 0.0
                for m in range(4):
                    tau = pheromones.get(m, node, cid)
                    eta = max(1e-12, etas[m])
                    log_score += float(weights[m]) * (
                        alpha * math.log(tau) + beta * math.log(eta)
                    )
                scores.append(log_score)
            max_log = max(scores)
            probs = np.exp(np.array(scores, dtype=float) - max_log)
            total = float(probs.sum())
            if total <= 0 or not math.isfinite(total):
                probs = np.ones(len(candidates), dtype=float) / len(candidates)
            else:
                probs = probs / total
            pick = int(rng.choice(len(candidates), p=probs))
            cid = candidates[pick]
            customer = ctx.instance.customers[cid]
            arrival = current_time + ctx.travel.travel_time(node, cid, current_time)
            start = max(arrival, float(customer.release_time), float(customer.ready_time))
            current_time = start + float(customer.service_time)
            load += float(customer.demand)
            seq.append(cid)
            unassigned.remove(cid)
            node = cid
            if not is_last and unassigned and rng.random() < stop_prob:
                break
        if seq:
            routes[vid] = seq

    return repair_routes(routes, ctx, rng)


def _update_pheromones(
    pheromones: PheromoneSet,
    archive: ParetoArchive,
    ctx: PlanningContext,
    cfg: PlannerConfig,
    rng: np.random.Generator,
) -> None:
    pheromones.evaporate(cfg.pheromone_evaporation)
    solutions = archive.solutions
    if not solutions:
        return
    seen_keys: Set[Tuple] = set()
    # 每个目标只由该目标上的档案最优解更新自己的信息素矩阵。
    for m in range(4):
        best = min(solutions, key=lambda s: (s.objectives[m], s.route_key()))
        pheromones.deposit_solution(best, ctx, 2.0, objectives=[m])
        seen_keys.add(best.route_key())
    # 稀疏区域解以较小权重补充到四组矩阵，维持探索而非让单点垄断。
    extra_count = 0
    while extra_count < 3:
        extra = archive.sample_leader(rng)
        if extra is None or extra.route_key() in seen_keys:
            break
        pheromones.deposit_solution(extra, ctx, 0.5)
        seen_keys.add(extra.route_key())
        extra_count += 1


def run_moaco(
    ctx: PlanningContext,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    warm_routes: Optional[Sequence[Dict[int, List[int]]]] = None,
) -> AlgorithmResult:
    started = time.perf_counter()
    budget = EvaluationBudget(cfg.max_evaluations)
    archive = ParetoArchive(max_size=cfg.archive_size, feasible_only=True)
    all_seen: List[Solution] = []
    pheromones = PheromoneSet(objectives=4)
    alpha = float(cfg.extra.get("alpha", 1.0))
    beta = float(cfg.extra.get("beta", 1.0))

    # 历史迁移：把上一事件方案作为先验沉积，但不绕过重新评价。
    if cfg.response_strategy != "restart" and warm_routes:
        for routes in list(warm_routes)[:3]:
            try:
                sol = budget.evaluate(repair_routes(routes, ctx, rng), ctx)
            except BudgetExhausted:
                break
            all_seen.append(sol)
            if sol.feasible:
                archive.add(sol)
                pheromones.deposit_solution(sol, ctx, 3.0)

    iterations = 0
    while budget.remaining > 0:
        iterations += 1
        n_ants = min(max(4, int(cfg.population_size)), budget.remaining)
        for _ in range(n_ants):
            weights = rng.dirichlet(np.ones(4))
            routes = _construct_ant(
                ctx, rng, pheromones, weights, alpha, beta,
                stop_probability=float(cfg.extra.get("route_stop_probability", 0.25)),
            )
            try:
                sol = budget.evaluate(routes, ctx)
            except BudgetExhausted:
                break
            all_seen.append(sol)
            if sol.feasible:
                archive.add(sol)
        _update_pheromones(pheromones, archive, ctx, cfg, rng)

    best_inf = best_infeasible(all_seen)
    return make_result(
        "moaco",
        archive.solutions,
        best_inf,
        budget.evaluations,
        started,
        cfg,
        {"iterations": iterations, "archive_size": len(archive)},
    )

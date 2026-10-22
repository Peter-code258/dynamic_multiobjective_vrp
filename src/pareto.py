"""Pareto 支配、快速非支配排序、拥挤距离与外部档案。"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import Solution


EPS = 1e-9


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """最小化意义下的 Pareto 支配。"""
    if len(a) != len(b):
        raise ValueError("目标维数不一致")
    not_worse = True
    strictly_better = False
    for x, y in zip(a, b):
        if x > y + EPS:
            not_worse = False
            break
        if x < y - EPS:
            strictly_better = True
    return not_worse and strictly_better


def constrained_dominates(a: Solution, b: Solution) -> bool:
    """约束支配：可行解优先；不可行解按违反度；可行解之间按 Pareto 支配。"""
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.violation_score < b.violation_score - EPS
    return dominates(a.objectives, b.objectives)


def fast_non_dominated_sort(solutions: Sequence[Solution]) -> List[List[int]]:
    """返回按前沿分层的索引列表（仅按目标支配，不考虑可行性）。"""
    n = len(solutions)
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(p + 1, n):
            if dominates(solutions[p].objectives, solutions[q].objectives):
                dominated_by[p].append(q)
                dom_count[q] += 1
            elif dominates(solutions[q].objectives, solutions[p].objectives):
                dominated_by[q].append(p)
                dom_count[p] += 1
    for i in range(n):
        if dom_count[i] == 0:
            fronts[0].append(i)
    idx = 0
    while idx < len(fronts) and fronts[idx]:
        nxt: List[int] = []
        for p in fronts[idx]:
            for q in dominated_by[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        idx += 1
        if nxt:
            fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(solutions: Sequence[Solution], indices: Sequence[int]) -> Dict[int, float]:
    """按前沿内目标归一化计算拥挤距离；边界解为无穷大。"""
    if not indices:
        return {}
    distance = {int(i): 0.0 for i in indices}
    if len(indices) <= 2:
        for i in indices:
            distance[int(i)] = float("inf")
        return distance
    dim = len(solutions[indices[0]].objectives)
    for m in range(dim):
        ordered = sorted(indices, key=lambda i: (solutions[i].objectives[m], i))
        lo = solutions[ordered[0]].objectives[m]
        hi = solutions[ordered[-1]].objectives[m]
        span = hi - lo
        distance[int(ordered[0])] = float("inf")
        distance[int(ordered[-1])] = float("inf")
        if span <= EPS:
            continue
        for k in range(1, len(ordered) - 1):
            prev_v = solutions[ordered[k - 1]].objectives[m]
            next_v = solutions[ordered[k + 1]].objectives[m]
            distance[int(ordered[k])] += (next_v - prev_v) / span
    return distance


def constrained_rank_and_crowding(
    solutions: Sequence[Solution],
) -> Tuple[List[int], List[float]]:
    """带约束的 NSGA-II 排序。

    可行解按 Pareto 前沿分层并排在前；不可行解按违反度排序排在后面，
    同违反度内再按目标支配分层。
    """
    n = len(solutions)
    ranks = [0] * n
    crowd = [0.0] * n
    feasible_idx = [i for i, s in enumerate(solutions) if s.feasible]
    infeasible_idx = [i for i, s in enumerate(solutions) if not s.feasible]
    rank_offset = 0

    if feasible_idx:
        sub = [solutions[i] for i in feasible_idx]
        fronts = fast_non_dominated_sort(sub)
        for r, front in enumerate(fronts):
            local = [feasible_idx[i] for i in front]
            cd = crowding_distance(solutions, local)
            for i in local:
                ranks[i] = r
                crowd[i] = cd.get(i, 0.0)
        rank_offset = len(fronts)

    if infeasible_idx:
        # 违反度相近的解视作同一层，再按目标非支配分层。
        infeasible_idx.sort(key=lambda i: (solutions[i].violation_score, i))
        levels: List[List[int]] = []
        current_score: Optional[float] = None
        for i in infeasible_idx:
            score = solutions[i].violation_score
            if current_score is None or abs(score - current_score) <= EPS:
                if not levels:
                    levels.append([])
                levels[-1].append(i)
            else:
                levels.append([i])
            current_score = score
        for level in levels:
            sub = [solutions[i] for i in level]
            fronts = fast_non_dominated_sort(sub)
            for r, front in enumerate(fronts):
                local = [level[i] for i in front]
                cd = crowding_distance(solutions, local)
                for i in local:
                    ranks[i] = rank_offset + r
                    crowd[i] = cd.get(i, 0.0)
            rank_offset += len(fronts)
    return ranks, crowd


class ParetoArchive:
    """外部非支配档案；默认只保存可行解。"""

    def __init__(self, max_size: int = 100, feasible_only: bool = True) -> None:
        self.max_size = int(max_size)
        self.feasible_only = feasible_only
        self._solutions: List[Solution] = []
        self._keys: set = set()

    def __len__(self) -> int:
        return len(self._solutions)

    @property
    def solutions(self) -> List[Solution]:
        return list(self._solutions)

    def _key(self, s: Solution) -> Tuple:
        return (
            tuple(round(float(x), 9) for x in s.objectives),
            s.feasible,
            round(float(s.violation_score), 6) if not s.feasible else 0.0,
        )

    def add(self, solution: Solution) -> bool:
        if self.feasible_only and not solution.feasible:
            return False
        key = self._key(solution)
        if key in self._keys:
            return False
        for existing in self._solutions:
            if dominates(existing.objectives, solution.objectives):
                return False
            if (
                all(abs(a - b) <= EPS for a, b in zip(existing.objectives, solution.objectives))
                and existing.route_key() == solution.route_key()
            ):
                return False
        kept: List[Solution] = []
        for existing in self._solutions:
            if not dominates(solution.objectives, existing.objectives):
                kept.append(existing)
        kept.append(solution)
        self._solutions = kept
        self._keys = {self._key(s) for s in self._solutions}
        if len(self._solutions) > self.max_size:
            self._prune()
        return True

    def _prune(self) -> None:
        if len(self._solutions) <= 2:
            return
        fronts = fast_non_dominated_sort(self._solutions)
        # 全部为同一前沿时使用拥挤距离删除最密集的解。
        indices = list(range(len(self._solutions)))
        cd = crowding_distance(self._solutions, indices)
        removable = [
            i for i in indices
            if math.isfinite(cd.get(i, float("inf")))
        ]
        if not removable:
            removable = indices[1:-1]
        if not removable:
            return
        worst = min(removable, key=lambda i: (cd.get(i, 0.0), i))
        self._solutions.pop(worst)
        self._keys = {self._key(s) for s in self._solutions}

    def sample_leader(self, rng: np.random.Generator) -> Optional[Solution]:
        if not self._solutions:
            return None
        if len(self._solutions) == 1:
            return self._solutions[0]
        indices = list(range(len(self._solutions)))
        cd = crowding_distance(self._solutions, indices)
        weights = np.array(
            [max(0.0, cd.get(i, 0.0)) if math.isfinite(cd.get(i, 0.0)) else 1e6 for i in indices],
            dtype=float,
        )
        total = float(weights.sum())
        if total <= 0:
            weights = np.ones(len(indices), dtype=float)
            total = float(weights.sum())
        pick = int(rng.choice(indices, p=weights / total))
        return self._solutions[pick]


def build_reference_front(archives: Sequence[Sequence[Solution]]) -> List[Solution]:
    """所有运行档案的合并非支配前沿（经验参考前沿，不是真实最优前沿）。"""
    union: List[Solution] = []
    for archive in archives:
        union.extend(archive)
    if not union:
        return []
    fronts = fast_non_dominated_sort(union)
    if not fronts:
        return []
    return [union[i] for i in fronts[0]]


def best_infeasible(solutions: Sequence[Solution]) -> Optional[Solution]:
    infeasible = [s for s in solutions if not s.feasible]
    if not infeasible:
        return None
    return min(
        infeasible,
        key=lambda s: (s.violation_score, sum(s.objectives), s.route_key()),
    )

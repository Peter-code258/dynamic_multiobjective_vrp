"""多目标解集质量指标：超体积 HV、IGD、IGD+，以及统一归一化。"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


Bounds = List[Tuple[float, float]]


def shared_bounds(points: Sequence[Sequence[float]], pad: float = 0.05) -> Bounds:
    arr = np.array(points, dtype=float)
    if arr.size == 0:
        return []
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return [(float(lo[i] - pad * span[i]), float(hi[i] + pad * span[i])) for i in range(arr.shape[1])]


def normalize_points(points: Sequence[Sequence[float]], bounds: Bounds) -> List[List[float]]:
    out: List[List[float]] = []
    for point in points:
        row = []
        for value, (lo, hi) in zip(point, bounds):
            span = hi - lo
            row.append(float((float(value) - lo) / span) if span > 1e-12 else 0.0)
        out.append(row)
    return out


def _hv(points: List[Tuple[float, ...]], ref: Tuple[float, ...]) -> float:
    if not points:
        return 0.0
    dim = len(ref)
    filtered = [p for p in points if all(p[i] < ref[i] for i in range(dim))]
    if not filtered:
        return 0.0
    # 必须按最后一维（切片方向）排序，才能保证每个 slab 内的投影是前缀集合。
    filtered = sorted(set(filtered), key=lambda p: p[-1])
    if dim == 1:
        return max(0.0, ref[0] - min(p[0] for p in filtered))
    volume = 0.0
    for i, point in enumerate(filtered):
        next_z = filtered[i + 1][-1] if i + 1 < len(filtered) else ref[-1]
        height = next_z - point[-1]
        if height <= 0:
            continue
        active = [p[:-1] for p in filtered[: i + 1]]
        volume += _hv(active, ref[:-1]) * height
    return volume


def hypervolume(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """最小化目标下的精确超体积（递归切片）。"""
    if not points:
        return 0.0
    return float(_hv([tuple(float(x) for x in p) for p in points], tuple(float(x) for x in ref)))


def igd(reference: Sequence[Sequence[float]], approx: Sequence[Sequence[float]]) -> float:
    if not reference or not approx:
        return float("nan")
    ref = np.array(reference, dtype=float)
    ap = np.array(approx, dtype=float)
    distances = []
    for r in ref:
        distances.append(float(np.min(np.sqrt(np.sum((ap - r) ** 2, axis=1)))))
    return float(np.mean(distances))


def igd_plus(reference: Sequence[Sequence[float]], approx: Sequence[Sequence[float]]) -> float:
    if not reference or not approx:
        return float("nan")
    ref = np.array(reference, dtype=float)
    ap = np.array(approx, dtype=float)
    distances = []
    for r in ref:
        diff = ap - r
        improved = np.maximum(0.0, diff)  # 近似解优于参考点时不计罚
        distances.append(float(np.min(np.sqrt(np.sum(improved ** 2, axis=1)))))
    return float(np.mean(distances))


def normalized_front_metrics(
    archives: Dict[str, Sequence[Sequence[float]]],
    reference_archives: Dict[str, Sequence[Sequence[float]]],
    all_points: Sequence[Sequence[float]],
    ref_point: Sequence[float] = (1.1, 1.1, 1.1, 1.1),
) -> Dict[str, Dict[str, float]]:
    """在同一归一化上下界下计算每个运行的 HV、IGD、IGD+。"""
    bounds = shared_bounds(all_points)
    if not bounds:
        return {}
    norm_archives = {k: normalize_points(v, bounds) for k, v in archives.items()}
    norm_refs = {k: normalize_points(v, bounds) for k, v in reference_archives.items()}
    # 经验参考前沿：所有运行档案的合并非支配集合
    flat_ref: List[Sequence[float]] = []
    for values in reference_archives.values():
        flat_ref.extend(values)
    if flat_ref:
        from .pareto import Solution, fast_non_dominated_sort

        dummy = [
            Solution(routes={}, objectives=tuple(p), feasible=True)
            for p in flat_ref
        ]
        fronts = fast_non_dominated_sort(dummy)
        empirical = [flat_ref[i] for i in fronts[0]] if fronts else flat_ref
    else:
        empirical = flat_ref
    norm_empirical = normalize_points(empirical, bounds)

    out: Dict[str, Dict[str, float]] = {}
    for key, points in norm_archives.items():
        if not points:
            out[key] = {"hv": 0.0, "igd": float("nan"), "igd_plus": float("nan")}
            continue
        out[key] = {
            "hv": hypervolume(points, ref_point),
            "igd": igd(norm_empirical, points),
            "igd_plus": igd_plus(norm_empirical, points),
        }
    return out

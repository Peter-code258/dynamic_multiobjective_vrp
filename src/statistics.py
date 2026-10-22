"""描述统计与非参数检验（Wilcoxon、Friedman、Holm 校正）。"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"),
            "median": float("nan"), "q25": float("nan"), "q75": float("nan"),
            "iqr": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"),
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = 0.0
    if arr.size > 1:
        half = float(stats.t.ppf(0.975, arr.size - 1) * std / math.sqrt(arr.size))
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """配对 Wilcoxon 符号秩检验 + 秩二列效应量。"""
    pairs = [
        (float(x), float(y))
        for x, y in zip(a, b)
        if x is not None and y is not None
        and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return {"n_pairs": len(pairs), "statistic": float("nan"), "p_value": float("nan"),
                "effect_rank_biserial": float("nan")}
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    diff = x - y
    nonzero = diff[np.abs(diff) > 1e-12]
    if nonzero.size == 0:
        return {"n_pairs": len(pairs), "statistic": 0.0, "p_value": 1.0,
                "effect_rank_biserial": 0.0}
    try:
        result = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        p = float(result.pvalue)
        stat = float(result.statistic)
    except ValueError:
        p = float("nan")
        stat = float("nan")
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    denom = w_pos + w_neg
    effect = (w_pos - w_neg) / denom if denom > 0 else 0.0
    return {
        "n_pairs": len(pairs),
        "statistic": stat,
        "p_value": p,
        "effect_rank_biserial": float(effect),
    }


def holm_correction(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    adjusted = [1.0] * m
    order = sorted(range(m), key=lambda i: p_values[i])
    running = 0.0
    for rank, idx in enumerate(order):
        value = float(p_values[idx])
        if not math.isfinite(value):
            adjusted[idx] = float("nan")
            continue
        corrected = min(1.0, (m - rank) * value)
        running = max(running, corrected)
        adjusted[idx] = running
    return adjusted


def friedman_test(matrix: Sequence[Sequence[float]]) -> Dict[str, float]:
    arr = np.array(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_blocks": arr.shape[0] if arr.ndim == 2 else 0}
    # 所有算法在该指标上完全相同的区组无法提供秩信息，避免除零警告。
    column_ranges = arr.max(axis=0) - arr.min(axis=0)
    if np.all(column_ranges <= 1e-12):
        return {"statistic": float("nan"), "p_value": float("nan"), "n_blocks": int(arr.shape[0])}
    try:
        stat, p = stats.friedmanchisquare(*[arr[:, j] for j in range(arr.shape[1])])
        return {"statistic": float(stat), "p_value": float(p), "n_blocks": int(arr.shape[0])}
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_blocks": int(arr.shape[0])}


def pairwise_wilcoxon_table(
    df: pd.DataFrame,
    metric: str,
    algorithms: Sequence[str],
    block_cols: Sequence[str] = ("instance_id", "scenario_id", "seed"),
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(index=list(block_cols), columns="algorithm", values=metric, aggfunc="mean")
    for a, b in itertools.combinations(list(algorithms), 2):
        if a not in pivot.columns or b not in pivot.columns:
            continue
        block = pivot[[a, b]].dropna()
        res = paired_wilcoxon(block[a].tolist(), block[b].tolist())
        rows.append(
            {
                "metric": metric,
                "algorithm_a": a,
                "algorithm_b": b,
                "n_pairs": res["n_pairs"],
                "wilcoxon_statistic": res["statistic"],
                "p_value": res["p_value"],
                "effect_rank_biserial": res["effect_rank_biserial"],
            }
        )
    if not rows:
        return pd.DataFrame()
    adjusted = holm_correction([r["p_value"] for r in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_value_holm"] = p_adj
        row["significant_0.05"] = bool(math.isfinite(p_adj) and p_adj < 0.05)
    return pd.DataFrame(rows)


def friedman_table(
    df: pd.DataFrame,
    metric: str,
    algorithms: Sequence[str],
    block_cols: Sequence[str] = ("instance_id", "scenario_id", "seed"),
) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(index=list(block_cols), columns="algorithm", values=metric, aggfunc="mean")
    cols = [a for a in algorithms if a in pivot.columns]
    if len(cols) < 3:
        return pd.DataFrame()
    pivot = pivot[cols].dropna()
    if len(pivot) < 3:
        return pd.DataFrame()
    result = friedman_test(pivot[cols].values.tolist())
    # 每个区组内跨算法排名，再对每个算法求平均秩。
    ranks = pivot[cols].rank(axis=1, ascending=True)
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "n_blocks": result["n_blocks"],
                "friedman_statistic": result["statistic"],
                "p_value": result["p_value"],
                **{"mean_rank_%s" % c: float(ranks[c].mean()) for c in cols},
            }
        ]
    )


def descriptive_table(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if df.empty:
        return pd.DataFrame()
    for keys, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric not in group.columns:
                continue
            stats_dict = descriptive_stats(group[metric].tolist())
            row = dict(base)
            row["metric"] = metric
            row.update(stats_dict)
            rows.append(row)
    return pd.DataFrame(rows)

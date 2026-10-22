"""实验图表输出（无界面后端，英文标签避免中文字体缺失）。"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = plt.get_cmap("tab10").colors


def _save(fig, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_routes(
    ax,
    customers: Dict[str, dict],
    routes: Dict[str, List[int]],
    depot: tuple,
    label_prefix: str = "",
) -> None:
    ax.scatter([depot[0]], [depot[1]], marker="*", s=220, c="black", label="Depot", zorder=5)
    for vid, route in routes.items():
        if not route:
            continue
        color = COLORS[int(vid) % len(COLORS)]
        xs = [depot[0]]
        ys = [depot[1]]
        for cid in route:
            item = customers.get(str(cid))
            if item is None:
                continue
            xs.append(item["x"])
            ys.append(item["y"])
        xs.append(depot[0])
        ys.append(depot[1])
        ax.plot(xs, ys, "-o", color=color, linewidth=1.5, markersize=4,
                label="%svehicle %s" % (label_prefix, vid))


def plot_route_map(
    customers: Dict[str, dict],
    routes: Dict[str, List[int]],
    path: str,
    depot: tuple = (0.0, 0.0),
    title: str = "Final routes",
    vehicles: Dict[str, dict] = None,
) -> None:
    """绘制最终状态路线图。

    图例区分：已完成客户（蓝点）、计划中路线（彩色连线）、启动时新增订单
    （绿色空心三角）、未服务客户（红色叉）与故障车辆（黑色叉）。
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _draw_routes(ax, customers, routes, depot, label_prefix="Planned: ")
    served = [c for c in customers.values() if c.get("status") == "SERVED"]
    unserved = [c for c in customers.values() if c.get("status") == "UNSERVED"]
    new_orders = [
        c for c in customers.values()
        if int(c.get("customer_id", 0)) >= 1001 and c.get("status") != "SERVED"
    ]
    if served:
        ax.scatter([c["x"] for c in served], [c["y"] for c in served],
                   s=35, c="#4c72b0", alpha=0.6, zorder=3, label="Completed")
    if new_orders:
        ax.scatter([c["x"] for c in new_orders], [c["y"] for c in new_orders],
                   s=70, facecolors="none", edgecolors="green", marker="^",
                   linewidths=1.4, zorder=4, label="New order (planned)")
    if unserved:
        ax.scatter([c["x"] for c in unserved], [c["y"] for c in unserved],
                   s=45, c="red", marker="x", zorder=4, label="Unserved")
    if vehicles:
        broken = [v for v in vehicles.values() if v.get("status") == "BROKEN"]
        for v in broken:
            node = str(v.get("current_node"))
            item = customers.get(node)
            if item is None:
                continue
            ax.scatter([item["x"]], [item["y"]], s=150, c="black", marker="X",
                       zorder=6, label="Broken vehicle")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_route_comparison(
    customers: Dict[str, dict],
    before: Dict[str, List[int]],
    after: Dict[str, List[int]],
    path: str,
    depot: tuple = (0.0, 0.0),
    title: str = "Before / after replanning",
    affected: Sequence[int] = (),
) -> None:
    """故障等事件前后的路线对比（md 12.5）：左为事件前计划，右为事件后计划。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _draw_routes(axes[0], customers, before, depot, label_prefix="Before: ")
    _draw_routes(axes[1], customers, after, depot, label_prefix="After: ")
    served = [c for c in customers.values() if c.get("status") == "SERVED"]
    unserved = [c for c in customers.values() if c.get("status") == "UNSERVED"]
    for ax in axes:
        if served:
            ax.scatter([c["x"] for c in served], [c["y"] for c in served],
                       s=22, c="#999999", alpha=0.5, zorder=2, label="Completed")
        if unserved:
            ax.scatter([c["x"] for c in unserved], [c["y"] for c in unserved],
                       s=45, c="red", marker="x", zorder=4, label="Unserved")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7)
    released = [customers[str(cid)] for cid in affected if str(cid) in customers]
    if released:
        axes[1].scatter([c["x"] for c in released], [c["y"] for c in released],
                        s=140, facecolors="none", edgecolors="orange", linewidths=2,
                        zorder=6, label="Released by event")
        axes[1].legend(loc="best", fontsize=7)
    axes[0].set_title("Before event")
    axes[1].set_title("After replanning")
    fig.suptitle(title)
    _save(fig, path)


def plot_pareto_projections(
    archives: Dict[str, Sequence[Sequence[float]]],
    path: str,
    title: str = "Pareto projections",
) -> None:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    names = ["f1 distance", "f2 vehicles", "f3 response", "f4 load var"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (i, j) in zip(axes.flatten(), pairs):
        for k, (algo, points) in enumerate(archives.items()):
            if not points:
                continue
            arr = np.array(points, dtype=float)
            ax.scatter(arr[:, i], arr[:, j], s=22, alpha=0.65, color=COLORS[k % len(COLORS)], label=algo)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.grid(alpha=0.25)
    axes.flatten()[0].legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, path)


def plot_metric_boxplot(df: pd.DataFrame, metric: str, path: str, title: str = "") -> None:
    if df.empty or metric not in df.columns:
        return
    algorithms = list(df["algorithm"].unique())
    data = [df[df["algorithm"] == a][metric].dropna().values for a in algorithms]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(data, labels=algorithms, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)


def plot_metric_lines(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    path: str,
    title: str = "",
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for idx, (name, group) in enumerate(df.groupby(hue_col)):
        grouped = group.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index()
        ax.plot(grouped[x_col], grouped["mean"], "-o", color=COLORS[idx % len(COLORS)], label=str(name))
        if grouped["std"].notna().any():
            ax.fill_between(
                grouped[x_col],
                grouped["mean"] - grouped["std"].fillna(0),
                grouped["mean"] + grouped["std"].fillna(0),
                color=COLORS[idx % len(COLORS)],
                alpha=0.15,
            )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or y_col)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_grouped_bars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    path: str,
    title: str = "",
) -> None:
    if df.empty:
        return
    table = df.pivot_table(index=x_col, columns=hue_col, values=y_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    table.plot(kind="bar", ax=ax, color=[COLORS[i % len(COLORS)] for i in range(len(table.columns))])
    ax.set_ylabel(y_col)
    ax.set_title(title or y_col)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    _save(fig, path)

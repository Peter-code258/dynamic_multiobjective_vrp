"""距离矩阵与时变行驶时间。

- 距离矩阵只由坐标决定；拥堵默认只改变行驶时间，不改变距离。
- 行驶时间 tau(i, j, t) = d_ij / speed * multiplier(i, j, t)。
- 同一时刻有多条拥堵时段覆盖同一条边时，取最大倍率，避免倍率相乘造成非物理膨胀。
- 时间函数按出发时刻取值，属于分段常数 FIFO 近似；仿真器按步长推进。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .models import VRPInstance


Edge = Tuple[int, int]


def normalize_edge(i: int, j: int) -> Edge:
    return (i, j) if i <= j else (j, i)


@dataclass
class CongestionPeriod:
    period_id: str
    start: float
    end: float
    edges: List[Edge]
    multiplier: float
    metadata: Dict[str, object] = None  # type: ignore[assignment]

    def active(self, t: float) -> bool:
        return self.start <= t < self.end


class TravelNetwork:
    """基于坐标的完全图路网。

    节点 ID 包括仓库 0 与所有客户 ID；动态新增客户在事件发生时调用 add_node。
    """

    def __init__(
        self,
        instance: VRPInstance,
        speed: float = 1.0,
    ) -> None:
        self.speed = float(speed)
        if self.speed <= 0:
            raise ValueError("speed 必须为正数")
        self.coords: Dict[int, Tuple[float, float]] = {
            instance.depot_id: (float(instance.depot_x), float(instance.depot_y))
        }
        for cid, customer in instance.customers.items():
            self.coords[cid] = (float(customer.x), float(customer.y))
        self._distance_cache: Dict[Edge, float] = {}
        self.periods: List[CongestionPeriod] = []

    def add_node(self, node_id: int, x: float, y: float) -> None:
        if node_id in self.coords:
            raise ValueError("重复节点 ID: %s" % node_id)
        self.coords[node_id] = (float(x), float(y))
        # 新节点加入后，旧的缓存仍然对旧节点对有效；新节点对在需要时计算。
        for other in list(self.coords.keys()):
            if other == node_id:
                continue
            edge = normalize_edge(node_id, other)
            if edge in self._distance_cache:
                continue

    def distance(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        edge = normalize_edge(i, j)
        cached = self._distance_cache.get(edge)
        if cached is not None:
            return cached
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        d = math.hypot(xi - xj, yi - yj)
        self._distance_cache[edge] = d
        return d

    def travel_time(self, i: int, j: int, departure_time: float) -> float:
        """返回从 i 出发、出发时刻为 departure_time 的行驶时间（分钟）。"""
        if i == j:
            return 0.0
        base = self.distance(i, j) / self.speed
        mult = self.active_multiplier(i, j, departure_time)
        return max(0.0, base * mult)

    def active_multiplier(self, i: int, j: int, t: float) -> float:
        edge = normalize_edge(i, j)
        active = [
            p.multiplier for p in self.periods if p.active(t) and edge in p.edges
        ]
        return max(active) if active else 1.0

    def add_congestion(
        self,
        period_id: str,
        start: float,
        duration: float,
        edges: Sequence[Sequence[int]],
        multiplier: float,
    ) -> None:
        if multiplier < 1.0:
            raise ValueError("拥堵倍率应不小于 1.0（不应把拥堵建模为加速）")
        normalized = [normalize_edge(int(e[0]), int(e[1])) for e in edges]
        self.periods.append(
            CongestionPeriod(
                period_id=period_id,
                start=float(start),
                end=float(start + duration),
                edges=normalized,
                multiplier=float(multiplier),
                metadata={},
            )
        )

    def remove_congestion(self, period_id: str) -> None:
        self.periods = [p for p in self.periods if p.period_id != period_id]

    def congestion_snapshot(self) -> List[Dict[str, object]]:
        return [
            {
                "period_id": p.period_id,
                "start": p.start,
                "end": p.end,
                "multiplier": p.multiplier,
                "edge_count": len(p.edges),
            }
            for p in self.periods
        ]

    def all_edges(self) -> List[Edge]:
        nodes = sorted(self.coords.keys())
        edges: List[Edge] = []
        for idx, i in enumerate(nodes):
            for j in nodes[idx + 1 :]:
                edges.append((i, j))
        return edges

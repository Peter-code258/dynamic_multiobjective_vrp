"""统一规划上下文。

算法只读取该上下文，不直接修改仿真实体；评估器与约束检查器共享同一份口径。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .models import VRPInstance, Vehicle
from .travel_time import TravelNetwork


@dataclass
class VehicleView:
    vehicle_id: int
    capacity: float
    start_node: int
    start_time: float
    available: bool
    cumulative_distance: float
    cumulative_demand: float
    cumulative_response: float


@dataclass
class PlanningContext:
    instance: VRPInstance
    travel: TravelNetwork
    remaining: List[int]
    vehicles: Dict[int, VehicleView]
    used_vehicle_ids: Set[int]
    max_route_duration: Optional[float] = None
    hard_time_windows: bool = False

    @classmethod
    def from_state(
        cls,
        instance: VRPInstance,
        travel: TravelNetwork,
        remaining: List[int],
        used_vehicle_ids: Set[int],
        max_route_duration: Optional[float] = None,
        hard_time_windows: bool = False,
        vehicles: Optional[Dict[int, Vehicle]] = None,
    ) -> "PlanningContext":
        source_vehicles = vehicles if vehicles is not None else instance.vehicles
        views: Dict[int, VehicleView] = {}
        for vid, v in source_vehicles.items():
            views[vid] = VehicleView(
                vehicle_id=vid,
                capacity=float(v.capacity),
                start_node=int(v.current_node),
                start_time=float(v.current_time),
                available=bool(v.available and v.status != "BROKEN"),
                cumulative_distance=float(v.cumulative_distance),
                cumulative_demand=float(v.cumulative_demand),
                cumulative_response=float(v.cumulative_response),
            )
        return cls(
            instance=instance,
            travel=travel,
            remaining=sorted(set(int(c) for c in remaining)),
            vehicles=views,
            used_vehicle_ids=set(int(v) for v in used_vehicle_ids),
            max_route_duration=max_route_duration,
            hard_time_windows=hard_time_windows,
        )

    @property
    def available_vehicle_ids(self) -> List[int]:
        return sorted(vid for vid, v in self.vehicles.items() if v.available)


def remaining_capacity(view: VehicleView) -> float:
    """车辆在本次规划中还可装载的需求量（扣除已服务订单）。"""
    return max(0.0, float(view.capacity) - float(view.cumulative_demand))

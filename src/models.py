"""核心数据模型。

本模块只描述静态数据与运行期状态，不包含任何优化算法逻辑。
所有目标函数、约束检查、算法与仿真器都复用这里的字段口径。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


UNRELEASED = "UNRELEASED"
PENDING = "PENDING"
ASSIGNED = "ASSIGNED"
SERVED = "SERVED"
UNSERVED = "UNSERVED"

IDLE = "IDLE"
ACTIVE = "ACTIVE"
BROKEN = "BROKEN"
FINISHED = "FINISHED"


@dataclass
class Customer:
    """客户/订单。

    时间单位统一为分钟；distance_unit 只是数据标签，算法按欧氏距离数值计算。
    release_time 是订单释放时刻；response_time = service_start - release_time。
    """

    customer_id: int
    x: float
    y: float
    demand: float
    release_time: float = 0.0
    service_time: float = 0.0
    ready_time: float = 0.0
    due_time: Optional[float] = None
    status: str = PENDING
    assigned_vehicle_id: Optional[int] = None
    service_start_time: Optional[float] = None
    arrival_time: Optional[float] = None

    def copy(self) -> "Customer":
        return Customer(
            customer_id=self.customer_id,
            x=self.x,
            y=self.y,
            demand=self.demand,
            release_time=self.release_time,
            service_time=self.service_time,
            ready_time=self.ready_time,
            due_time=self.due_time,
            status=self.status,
            assigned_vehicle_id=self.assigned_vehicle_id,
            service_start_time=self.service_start_time,
            arrival_time=self.arrival_time,
        )


@dataclass
class Vehicle:
    """车辆静态属性 + 仿真运行期状态。

    available=False 表示车辆故障或不可用；route_suffix 只保存尚未执行的客户序列。
    cumulative_* 字段记录已实际执行的部分，重规划不得回滚。
    """

    vehicle_id: int
    capacity: float
    start_time: float = 0.0
    status: str = IDLE
    current_node: int = 0
    current_time: float = 0.0
    route_suffix: List[int] = field(default_factory=list)
    served_customers: List[int] = field(default_factory=list)
    cumulative_distance: float = 0.0
    cumulative_response: float = 0.0
    cumulative_demand: float = 0.0
    used: bool = False
    available: bool = True

    def copy(self) -> "Vehicle":
        return Vehicle(
            vehicle_id=self.vehicle_id,
            capacity=self.capacity,
            start_time=self.start_time,
            status=self.status,
            current_node=self.current_node,
            current_time=self.current_time,
            route_suffix=list(self.route_suffix),
            served_customers=list(self.served_customers),
            cumulative_distance=self.cumulative_distance,
            cumulative_response=self.cumulative_response,
            cumulative_demand=self.cumulative_demand,
            used=self.used,
            available=self.available,
        )


@dataclass
class VRPInstance:
    instance_id: str
    depot_id: int
    depot_x: float
    depot_y: float
    vehicles: Dict[int, Vehicle]
    customers: Dict[int, Customer]
    distance_type: str = "euclidean"
    time_unit: str = "minute"
    distance_unit: str = "distance_unit"
    source: str = "synthetic"
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def customer_ids(self) -> List[int]:
        return sorted(self.customers.keys())

    @property
    def vehicle_ids(self) -> List[int]:
        return sorted(self.vehicles.keys())

    def add_customer(self, customer: Customer) -> None:
        if customer.customer_id in self.customers:
            raise ValueError("重复的 customer_id: %s" % customer.customer_id)
        self.customers[customer.customer_id] = customer

    def depth_copy(self) -> "VRPInstance":
        return VRPInstance(
            instance_id=self.instance_id,
            depot_id=self.depot_id,
            depot_x=self.depot_x,
            depot_y=self.depot_y,
            vehicles={vid: v.copy() for vid, v in self.vehicles.items()},
            customers={cid: c.copy() for cid, c in self.customers.items()},
            distance_type=self.distance_type,
            time_unit=self.time_unit,
            distance_unit=self.distance_unit,
            source=self.source,
            metadata=dict(self.metadata),
        )


@dataclass
class DynamicEvent:
    event_id: str
    event_type: str
    event_time: float
    payload: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id,
            "type": self.event_type,
            "time": self.event_time,
            "payload": self.payload,
        }


@dataclass
class Scenario:
    scenario_id: str
    seed: int
    events: List[DynamicEvent]
    intensity: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "intensity": dict(self.intensity),
            "metadata": dict(self.metadata),
            "events": [
                {
                    "event_id": e.event_id,
                    "type": e.event_type,
                    "time": e.event_time,
                    **dict(e.payload),
                }
                for e in self.events
            ],
        }


@dataclass
class Evaluation:
    """统一评估器输出。objectives 顺序固定为 [f1, f2, f3, f4]。"""

    objectives: Tuple[float, float, float, float]
    feasible: bool
    violations: List[str] = field(default_factory=list)
    unserved: List[int] = field(default_factory=list)
    violation_score: float = 0.0
    vehicle_details: Dict[int, Dict[str, object]] = field(default_factory=dict)
    total_lateness: float = 0.0
    total_early_wait: float = 0.0
    served_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "objectives": list(self.objectives),
            "feasible": self.feasible,
            "violations": list(self.violations),
            "unserved": list(self.unserved),
            "violation_score": self.violation_score,
            "vehicle_details": self.vehicle_details,
            "total_lateness": self.total_lateness,
            "total_early_wait": self.total_early_wait,
            "served_count": self.served_count,
        }


@dataclass
class Solution:
    """一条带评估结果的候选方案。routes 的键为车辆 ID，值只含客户序列。"""

    routes: Dict[int, List[int]]
    objectives: Tuple[float, ...] = ()
    feasible: bool = False
    violation_score: float = float("inf")
    violations: List[str] = field(default_factory=list)
    evaluation: Optional[Evaluation] = None

    def copy(self) -> "Solution":
        return Solution(
            routes={vid: list(route) for vid, route in self.routes.items()},
            objectives=tuple(self.objectives),
            feasible=self.feasible,
            violation_score=self.violation_score,
            violations=list(self.violations),
            evaluation=self.evaluation,
        )

    def route_key(self) -> Tuple:
        return tuple(
            (vid, tuple(route)) for vid, route in sorted(self.routes.items()) if route
        )

    def to_dict(self, include_routes: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "objectives": [float(x) for x in self.objectives],
            "feasible": self.feasible,
            "violation_score": self.violation_score,
            "violations": list(self.violations),
        }
        if include_routes:
            out["routes"] = {str(vid): list(r) for vid, r in sorted(self.routes.items()) if r}
        return out

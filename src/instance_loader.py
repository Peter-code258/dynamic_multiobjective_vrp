"""实例与场景文件读取、校验。

支持两类实例：
1. 本项目约定的 JSON 格式（见 data/instances/*.json）。
2. Solomon VRPTW 文本格式（100 客户基准）；可截取前 N 个客户，派生实例需在元数据中标注。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .models import (
    Customer,
    DynamicEvent,
    PENDING,
    Scenario,
    UNRELEASED,
    VRPInstance,
    Vehicle,
)


class InstanceFormatError(ValueError):
    pass


def _require_keys(obj: dict, keys: List[str], where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise InstanceFormatError("%s 缺少字段: %s" % (where, ", ".join(missing)))


def load_instance(path: str) -> VRPInstance:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return load_json_instance(path)
    if ext in (".txt", ".sol", ".vrp"):
        return load_solomon_instance(path)
    raise InstanceFormatError("无法识别的实例格式: %s" % path)


def load_json_instance(path: str) -> VRPInstance:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    _require_keys(raw, ["instance_id", "depot", "vehicles", "customers"], "实例")
    depot = raw["depot"]
    _require_keys(depot, ["id", "x", "y"], "depot")

    vehicles: Dict[int, Vehicle] = {}
    for idx, item in enumerate(raw["vehicles"]):
        _require_keys(item, ["vehicle_id", "capacity"], "vehicles[%d]" % idx)
        vid = int(item["vehicle_id"])
        if vid in vehicles:
            raise InstanceFormatError("重复车辆 ID: %s" % vid)
        vehicles[vid] = Vehicle(
            vehicle_id=vid,
            capacity=float(item["capacity"]),
            start_time=float(item.get("start_time", 0.0)),
            current_node=int(depot["id"]),
            current_time=float(item.get("start_time", 0.0)),
            status="IDLE",
            available=bool(item.get("available", True)),
        )

    customers: Dict[int, Customer] = {}
    for idx, item in enumerate(raw["customers"]):
        _require_keys(
            item,
            ["customer_id", "x", "y", "demand", "service_time"],
            "customers[%d]" % idx,
        )
        cid = int(item["customer_id"])
        if cid in customers:
            raise InstanceFormatError("重复客户 ID: %s" % cid)
        release = float(item.get("release_time", 0.0))
        customers[cid] = Customer(
            customer_id=cid,
            x=float(item["x"]),
            y=float(item["y"]),
            demand=float(item["demand"]),
            release_time=release,
            service_time=float(item["service_time"]),
            ready_time=float(item.get("ready_time", release)),
            due_time=None if item.get("due_time") is None else float(item["due_time"]),
            status=PENDING if release <= 0.0 else UNRELEASED,
        )

    if not vehicles:
        raise InstanceFormatError("实例至少需要一辆车")
    if not customers:
        raise InstanceFormatError("实例至少需要一个客户")

    return VRPInstance(
        instance_id=str(raw["instance_id"]),
        depot_id=int(depot["id"]),
        depot_x=float(depot["x"]),
        depot_y=float(depot["y"]),
        vehicles=vehicles,
        customers=customers,
        distance_type=str(raw.get("distance_type", "euclidean")),
        time_unit=str(raw.get("time_unit", "minute")),
        distance_unit=str(raw.get("distance_unit", "distance_unit")),
        source=str(raw.get("source", "json")),
        metadata=dict(raw.get("metadata", {})),
    )


def load_solomon_instance(
    path: str,
    instance_id: Optional[str] = None,
    max_customers: Optional[int] = None,
    num_vehicles: Optional[int] = None,
) -> VRPInstance:
    """读取 Solomon 文本实例。

    只使用坐标、需求、时间窗和服务时长；车辆数取文件中声明值或参数覆盖。
    截取子集时 source 标记为派生实例，避免把原始最佳解误用于子实例。
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    header_capacity = None
    declared_vehicles = None
    data_lines: List[str] = []
    in_customer_section = False
    for line in lines:
        upper = line.upper()
        if upper.startswith("CUSTOMER"):
            in_customer_section = True
            continue
        if in_customer_section:
            parts = line.split()
            if len(parts) >= 7 and _is_number(parts[0]):
                data_lines.append(line)
            continue
        if upper.startswith("NUMBER"):
            continue
        parts = line.split()
        if len(parts) >= 2 and all(_is_number(p) for p in parts[:2]):
            declared_vehicles = int(float(parts[0]))
            header_capacity = float(parts[1])

    if header_capacity is None or not data_lines:
        raise InstanceFormatError("不是可识别的 Solomon 实例文件: %s" % path)

    rows = []
    for line in data_lines:
        parts = line.split()
        rows.append(
            {
                "id": int(float(parts[0])),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "demand": float(parts[3]),
                "ready": float(parts[4]),
                "due": float(parts[5]),
                "service": float(parts[6]),
            }
        )
    rows.sort(key=lambda r: r["id"])
    depot = rows[0]
    customers_raw = rows[1:]
    truncated = False
    if max_customers is not None and max_customers < len(customers_raw):
        customers_raw = customers_raw[:max_customers]
        truncated = True

    vehicle_count = int(num_vehicles or declared_vehicles or max(1, len(customers_raw)))
    vehicles = {
        vid: Vehicle(
            vehicle_id=vid,
            capacity=float(header_capacity),
            start_time=0.0,
            current_node=int(depot["id"]),
            current_time=0.0,
        )
        for vid in range(vehicle_count)
    }
    customers = {
        int(r["id"]): Customer(
            customer_id=int(r["id"]),
            x=float(r["x"]),
            y=float(r["y"]),
            demand=float(r["demand"]),
            # Solomon 的 ready/due 是时间窗，不是动态订单释放时刻；
            # 静态实例中的订单视为在 0 时刻已释放，否则 f3 会被错误地放大。
            release_time=0.0,
            service_time=float(r["service"]),
            ready_time=float(r["ready"]),
            due_time=float(r["due"]),
            status=PENDING,
        )
        for r in customers_raw
    }
    derived = truncated or num_vehicles is not None
    return VRPInstance(
        instance_id=instance_id or os.path.splitext(os.path.basename(path))[0],
        depot_id=int(depot["id"]),
        depot_x=float(depot["x"]),
        depot_y=float(depot["y"]),
        vehicles=vehicles,
        customers=customers,
        source="solomon-derived" if derived else "solomon",
        metadata={
            "original_file": os.path.basename(path),
            "original_customer_count": len(rows) - 1,
            "used_customer_count": len(customers_raw),
            "derived_subset": truncated,
        },
    )


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def load_scenario(path: str) -> Scenario:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    _require_keys(raw, ["scenario_id", "seed", "events"], "场景")
    events: List[DynamicEvent] = []
    for idx, item in enumerate(raw["events"]):
        if "type" not in item or "time" not in item:
            raise InstanceFormatError("events[%d] 缺少 type/time" % idx)
        payload = {
            k: v
            for k, v in item.items()
            if k not in ("event_id", "type", "time")
        }
        events.append(
            DynamicEvent(
                event_id=str(item.get("event_id", "e%03d" % idx)),
                event_type=str(item["type"]).upper(),
                event_time=float(item["time"]),
                payload=payload,
            )
        )
    events.sort(key=lambda e: (e.event_time, e.event_id))
    return Scenario(
        scenario_id=str(raw["scenario_id"]),
        seed=int(raw.get("seed", 0)),
        events=events,
        intensity=dict(raw.get("intensity", {})),
        metadata=dict(raw.get("metadata", {})),
    )


def save_instance(instance: VRPInstance, path: str) -> None:
    data = {
        "instance_id": instance.instance_id,
        "depot": {"id": instance.depot_id, "x": instance.depot_x, "y": instance.depot_y},
        "vehicles": [
            {"vehicle_id": v.vehicle_id, "capacity": v.capacity, "start_time": v.start_time}
            for v in instance.vehicles.values()
        ],
        "customers": [
            {
                "customer_id": c.customer_id,
                "x": c.x,
                "y": c.y,
                "demand": c.demand,
                "service_time": c.service_time,
                "release_time": c.release_time,
                "ready_time": c.ready_time,
                "due_time": c.due_time,
            }
            for c in instance.customers.values()
        ],
        "distance_type": instance.distance_type,
        "time_unit": instance.time_unit,
        "distance_unit": instance.distance_unit,
        "source": instance.source,
        "metadata": instance.metadata,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_scenario(scenario: Scenario, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenario.to_dict(), f, ensure_ascii=False, indent=2)

"""动态事件、受影响对象与恢复指标测试。"""
import pytest

from src.algorithms.base import PlannerConfig
from src.models import DynamicEvent, Scenario
from src.simulator import SimulationConfig, SimulationEngine


def _engine(instance, scenario, algorithm="nsga2", seed=5):
    return SimulationEngine(
        instance=instance,
        scenario=scenario,
        algorithm=algorithm,
        seed=seed,
        planner_config=PlannerConfig(population_size=10, max_evaluations=250),
        sim_config=SimulationConfig(horizon=200.0),
    )


def test_new_order_records_affected_customer_and_recovery(toy_instance):
    scenario = Scenario(
        scenario_id="S1",
        seed=1,
        events=[
            DynamicEvent(
                event_id="e1",
                event_type="NEW_ORDER",
                event_time=40.0,
                payload={
                    "customer": {
                        "customer_id": 1001,
                        "x": 15.0,
                        "y": 15.0,
                        "demand": 10.0,
                        "service_time": 1.0,
                        "release_time": 40.0,
                    }
                },
            )
        ],
    )
    result = _engine(toy_instance, scenario).run()
    replan = next(r for r in result["replans"] if r["event_id"] == "e1")
    assert replan["affected_customers"] == [1001]
    assert replan["available_vehicles"] == 2
    assert "vehicle_loads" in replan
    recovery = next(r for r in result["recovery"] if r["event_id"] == "e1")
    assert recovery["recovered"] is True
    assert recovery["recovery_time"] >= 0.0
    assert recovery["recovery_evaluations"] > 0
    assert result["metrics"]["recovery_rate"] == pytest.approx(1.0)


def test_breakdown_reports_released_customers_and_never_overloads(toy_instance):
    instance = toy_instance.depth_copy()
    from src.models import Vehicle

    for vehicle in instance.vehicles.values():
        vehicle.capacity = 40.0
    instance.vehicles[2] = Vehicle(vehicle_id=2, capacity=40.0)
    scenario = Scenario(
        scenario_id="S2",
        seed=1,
        events=[
            DynamicEvent(
                event_id="e1",
                event_type="VEHICLE_BREAKDOWN",
                event_time=5.0,
                payload={"vehicle_id": 0},
            )
        ],
    )
    engine = _engine(instance, scenario)
    result = engine.run()
    replan = next(r for r in result["replans"] if r["event_id"] == "e1")
    assert replan["affected_vehicles"] == [0]
    assert isinstance(replan["affected_customers"], list)
    assert engine.instance.vehicles[0].available is False
    assert engine.instance.vehicles[0].status == "BROKEN"
    for vid, snapshot in result["vehicles"].items():
        assert snapshot["cumulative_demand"] <= engine.instance.vehicles[int(vid)].capacity + 1e-9


def test_congestion_records_edges_and_network_version(toy_instance):
    scenario = Scenario(
        scenario_id="S3",
        seed=1,
        events=[
            DynamicEvent(
                event_id="e1",
                event_type="CONGESTION_START",
                event_time=10.0,
                payload={
                    "period_id": "c1",
                    "edges": [[0, 1], [0, 2]],
                    "travel_time_multiplier": 1.5,
                    "duration": 50.0,
                },
            )
        ],
    )
    result = _engine(toy_instance, scenario, algorithm="greedy").run()
    replan = next(r for r in result["replans"] if r["event_id"] == "e1")
    assert replan["affected_edges"] == [[0, 1], [0, 2]]
    assert replan["network_version"] == 1
    assert result["congestion_periods"][0]["period_id"] == "c1"


def test_combined_event_keeps_unserved_orders_explicit(toy_instance):
    """组合事件且容量不足时，未服务订单必须写入结果，不能静默丢弃。"""
    instance = toy_instance.depth_copy()
    for vehicle in instance.vehicles.values():
        vehicle.capacity = 20.0  # 总容量 40 < 初始需求 100
    scenario = Scenario(
        scenario_id="S4",
        seed=1,
        events=[
            DynamicEvent(
                event_id="e1",
                event_type="NEW_ORDER",
                event_time=30.0,
                payload={
                    "customer": {
                        "customer_id": 1001,
                        "x": 12.0,
                        "y": 12.0,
                        "demand": 10.0,
                        "service_time": 1.0,
                        "release_time": 30.0,
                    }
                },
            ),
            DynamicEvent(
                event_id="e2",
                event_type="VEHICLE_BREAKDOWN",
                event_time=50.0,
                payload={"vehicle_id": 1},
            ),
        ],
    )
    result = _engine(instance, scenario, algorithm="greedy").run()
    assert result["metrics"]["unserved_orders"] > 0
    assert result["metrics"]["service_rate"] < 1.0
    assert result["metrics"]["feasible"] is False
    unserved = [c for c in result["customers"].values() if c["status"] == "UNSERVED"]
    assert len(unserved) == result["metrics"]["unserved_orders"]
    assert result["metrics"]["unrecovered_events"] >= 0
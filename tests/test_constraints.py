"""统一约束检查器测试：结构、重复服务、累计容量与不可用车辆。"""
from src.context import PlanningContext
from src.constraints import check_routes
from src.travel_time import TravelNetwork


def test_feasible_routes_pass(context):
    result = check_routes({0: [1, 2], 1: [3, 4, 5]}, context)
    assert result.feasible
    assert result.unserved == []
    assert result.loads == {0: 40.0, 1: 60.0}


def test_missing_customer_is_reported(context):
    result = check_routes({0: [1, 2]}, context)
    assert not result.feasible
    assert result.unserved == [3, 4, 5]
    assert any("未服务客户" in v for v in result.violations)


def test_duplicate_customer_is_reported(context):
    result = check_routes({0: [1, 2], 1: [2, 3, 4, 5]}, context)
    assert not result.feasible
    assert any("被重复服务" in v for v in result.violations)


def test_cumulative_capacity_uses_served_demand(context):
    instance = context.instance.depth_copy()
    travel = TravelNetwork(instance)
    instance.vehicles[0].cumulative_demand = 50.0  # 已服务 50，容量 60，只剩 10
    ctx = PlanningContext.from_state(
        instance=instance,
        travel=travel,
        remaining=[1, 2, 3, 4, 5],
        used_vehicle_ids={0},
        vehicles=instance.vehicles,
    )
    result = check_routes({0: [1, 2], 1: [3, 4, 5]}, ctx)
    assert not result.feasible
    assert any("累计超载" in v for v in result.violations)
    assert result.violation_score > 0


def test_unknown_and_unavailable_vehicle(context):
    instance = context.instance.depth_copy()
    travel = TravelNetwork(instance)
    instance.vehicles[1].available = False
    instance.vehicles[1].status = "BROKEN"
    ctx = PlanningContext.from_state(
        instance=instance,
        travel=travel,
        remaining=[1, 2, 3, 4, 5],
        used_vehicle_ids=set(),
        vehicles=instance.vehicles,
    )
    result = check_routes({0: [1, 2], 1: [3, 4, 5], 99: [1]}, ctx)
    assert not result.feasible
    assert any("不可用" in v for v in result.violations)
    assert any("未知车辆" in v for v in result.violations)

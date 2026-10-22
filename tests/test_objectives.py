import pytest

from src.constraints import check_routes
from src.objectives import evaluate_plan


def test_feasible_routes_and_objectives(context):
    routes = {0: [1, 2, 3], 1: [4, 5]}
    check = check_routes(routes, context)
    assert check.feasible
    ev = evaluate_plan(routes, context)
    assert ev.feasible
    assert ev.served_count == 5
    assert ev.objectives[1] == 2.0
    assert ev.objectives[0] > 0
    assert ev.objectives[2] > 0
    assert ev.total_lateness == 0.0


def test_unserved_customer_is_infeasible(context):
    routes = {0: [1, 2]}
    ev = evaluate_plan(routes, context)
    assert not ev.feasible
    assert set(ev.unserved) == {3, 4, 5}
    assert ev.violation_score > 0


def test_duplicate_customer_is_infeasible(context):
    routes = {0: [1, 2, 1], 1: [3, 4, 5]}
    ev = evaluate_plan(routes, context)
    assert not ev.feasible
    assert any("重复" in v for v in ev.violations)
    assert ev.served_count == 5


def test_capacity_overload_is_infeasible(context):
    # 每条路线各 80，超过 60 容量
    routes = {0: [1, 2, 3, 4], 1: [5]}
    ev = evaluate_plan(routes, context)
    assert not ev.feasible
    assert any("超载" in v for v in ev.violations)


def test_unknown_customer_is_infeasible(context):
    routes = {0: [1, 2, 99], 1: [3, 4, 5]}
    ev = evaluate_plan(routes, context)
    assert not ev.feasible
    assert any("99" in v for v in ev.violations)


def test_f2_counts_used_vehicles_and_cumulative_history(context):
    context.used_vehicle_ids = {1}
    routes = {0: [1, 2, 3], 1: [4, 5]}
    ev = evaluate_plan(routes, context)
    assert ev.objectives[1] == 2.0


def test_release_time_increases_response(context, toy_instance):
    toy_instance.customers[1].release_time = 50.0
    toy_instance.customers[1].status = "PENDING"
    ev = evaluate_plan({0: [1, 2, 3], 1: [4, 5]}, context)
    assert ev.objectives[2] >= 50.0
    assert ev.feasible


def test_hard_time_window_violation(context, toy_instance):
    toy_instance.customers[1].due_time = 1.0
    context.hard_time_windows = True
    ev = evaluate_plan({0: [1, 2, 3], 1: [4, 5]}, context)
    assert not ev.feasible
    assert any("时间窗" in v for v in ev.violations)


def test_capacity_check_includes_already_served_demand(context):
    context.vehicles[0].cumulative_demand = 50.0
    assert context.vehicles[0].capacity == 60.0
    ev = evaluate_plan({0: [1], 1: [2, 3, 4, 5]}, context)
    assert not ev.feasible
    assert any("累计超载" in v for v in ev.violations)


def test_repair_respects_remaining_capacity(context):
    from src.solution import repair_routes, route_load

    context.vehicles[0].cumulative_demand = 50.0
    routes = repair_routes({0: [1, 2, 3], 1: [4, 5]}, context)
    assert route_load(routes.get(0, []), context) <= 10.0 + 1e-9
    assert route_load(routes.get(1, []), context) <= 60.0 + 1e-9

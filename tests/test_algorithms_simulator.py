import copy

import numpy as np
import pytest

from src.algorithms import get_planner
from src.algorithms.base import PlannerConfig, initial_route_seeds
from src.baselines import run_greedy
from src.models import DynamicEvent, Scenario
from src.pareto import dominates
from src.simulator import SimulationConfig, SimulationEngine
from src.solution import greedy_solution


@pytest.mark.parametrize("algorithm", ["nsga2", "mopso", "moaco"])
def test_algorithms_return_non_dominated_feasible_archive(algorithm, context):
    cfg = PlannerConfig(population_size=8, max_evaluations=160, archive_size=20)
    rng = np.random.default_rng(7)
    planner = get_planner(algorithm)
    result = planner(context, cfg, rng, [])
    assert result.evaluations <= cfg.max_evaluations
    assert all(s.feasible for s in result.archive)
    for i, a in enumerate(result.archive):
        for j, b in enumerate(result.archive):
            if i != j:
                assert not dominates(b.objectives, a.objectives)
    assert result.algorithm == algorithm


def test_greedy_baseline_is_feasible_and_deterministic(context):
    cfg = PlannerConfig(max_evaluations=10)
    r1 = run_greedy(context, cfg, np.random.default_rng(1), [])
    r2 = run_greedy(context, cfg, np.random.default_rng(2), [])
    assert r1.archive[0].objectives == r2.archive[0].objectives
    assert r1.archive[0].feasible
    assert r1.evaluations == 1


def test_response_strategies_control_history_injection(context):
    warm = greedy_solution(context, np.random.default_rng(1))
    restart_seeds = initial_route_seeds(
        context, PlannerConfig(response_strategy="restart"), np.random.default_rng(2), [warm], 5
    )
    warm_seeds = initial_route_seeds(
        context, PlannerConfig(response_strategy="warm_start"), np.random.default_rng(2), [warm], 5
    )
    mixed_seeds = initial_route_seeds(
        context, PlannerConfig(response_strategy="mixed"), np.random.default_rng(2), [warm], 5
    )
    assert all(seed != warm for seed in restart_seeds)
    assert warm_seeds[0] == warm
    assert mixed_seeds[0] == warm
    assert mixed_seeds[1] != warm


def test_moaco_archive_keeps_vehicle_count_tradeoff(toy_instance):
    """容量足够时也必须保留“少用车 vs 低响应”的折中，不能塌缩为单点。"""
    from src.context import PlanningContext
    from src.travel_time import TravelNetwork

    instance = toy_instance.depth_copy()
    instance.vehicles[0].capacity = 100.0
    instance.vehicles[1].capacity = 100.0
    ctx = PlanningContext.from_state(
        instance=instance,
        travel=TravelNetwork(instance),
        remaining=list(instance.customers.keys()),
        used_vehicle_ids=set(),
    )
    result = get_planner("moaco")(
        ctx,
        PlannerConfig(population_size=10, max_evaluations=400, archive_size=20),
        np.random.default_rng(7),
        [],
    )
    assert len(result.archive) > 1
    assert len({int(s.objectives[1]) for s in result.archive}) > 1


def _make_engine(toy_instance, scenario, algorithm="nsga2", seed=3, max_eval=200):
    return SimulationEngine(
        instance=toy_instance,
        scenario=scenario,
        algorithm=algorithm,
        seed=seed,
        planner_config=PlannerConfig(population_size=10, max_evaluations=max_eval),
        sim_config=SimulationConfig(horizon=200.0),
    )


def test_simulator_static_reproducible_and_complete(toy_instance):
    scenario = Scenario(scenario_id="S0", seed=1, events=[])
    e1 = _make_engine(toy_instance, scenario)
    r1 = e1.run()
    e2 = _make_engine(toy_instance, scenario)
    r2 = e2.run()
    assert r1["metrics"]["service_rate"] == pytest.approx(1.0)
    assert r1["metrics"]["feasible"]
    assert r1["metrics"]["f1_total_distance"] > 0
    assert r1["metrics"]["f1_total_distance"] == pytest.approx(
        r2["metrics"]["f1_total_distance"]
    )
    assert r1["metrics"]["f3_total_response_time"] == pytest.approx(
        r2["metrics"]["f3_total_response_time"]
    )
    assert [x["objectives"] for x in r1["replans"]] == [x["objectives"] for x in r2["replans"]]


def test_breakdown_event_is_applied_and_replanned(toy_instance):
    instance = toy_instance.depth_copy()
    for vehicle in instance.vehicles.values():
        vehicle.capacity = 40.0
    for vid in (2, 3):
        from src.models import Vehicle

        instance.vehicles[vid] = Vehicle(vehicle_id=vid, capacity=40.0)
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
    engine = _make_engine(instance, scenario, max_eval=300)
    result = engine.run()
    assert [r["event_type"] for r in result["replans"]] == ["INITIAL", "VEHICLE_BREAKDOWN"]
    assert engine.instance.vehicles[0].available is False
    assert engine.instance.vehicles[0].status == "BROKEN"
    assert result["metrics"]["service_rate"] == pytest.approx(1.0)


def test_new_order_event_adds_customer(toy_instance):
    scenario = Scenario(
        scenario_id="S1",
        seed=1,
        events=[
            DynamicEvent(
                event_id="e1",
                event_type="NEW_ORDER",
                event_time=50.0,
                payload={
                    "customer": {
                        "customer_id": 1001,
                        "x": 10.0,
                        "y": 10.0,
                        "demand": 10.0,
                        "service_time": 1.0,
                        "release_time": 50.0,
                    }
                },
            )
        ],
    )
    engine = _make_engine(toy_instance, scenario)
    result = engine.run()
    assert "1001" in result["customers"]
    assert result["metrics"]["released_orders"] == 6
    assert result["metrics"]["service_rate"] == pytest.approx(1.0)
    assert result["replans"][1]["event_type"] == "NEW_ORDER"


def test_congestion_event_changes_travel_time(toy_instance):
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
                    "edges": [[0, 1], [0, 2], [0, 3]],
                    "travel_time_multiplier": 2.0,
                    "duration": 100.0,
                },
            )
        ],
    )
    engine = _make_engine(toy_instance, scenario)
    engine.run()
    assert len(engine.travel.periods) == 1
    assert engine.travel.active_multiplier(0, 1, 50.0) == pytest.approx(2.0)
    assert engine.travel.active_multiplier(0, 1, 150.0) == pytest.approx(1.0)


def test_capacity_is_never_exceeded_even_when_demand_exceeds_fleet(toy_instance):
    instance = toy_instance.depth_copy()
    for vehicle in instance.vehicles.values():
        vehicle.capacity = 20.0  # 总容量 40 < 总需求 100
    scenario = Scenario(scenario_id="overload", seed=1, events=[])
    engine = _make_engine(instance, scenario, max_eval=200)
    result = engine.run()
    for vid, snapshot in result["vehicles"].items():
        assert snapshot["cumulative_demand"] <= engine.instance.vehicles[int(vid)].capacity + 1e-9
    assert result["metrics"]["service_rate"] < 1.0
    assert result["metrics"]["unserved_orders"] > 0

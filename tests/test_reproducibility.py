"""可复现性测试：事件日程、算法随机流与仿真结果。"""
import pytest

from src.algorithms.base import PlannerConfig
from src.models import Scenario
from src.scenario_generator import generate_scenario, scenario_seed
from src.simulator import SimulationConfig, SimulationEngine


def _scenario_dict(scenario):
    return [
        (e.event_id, e.event_type, e.event_time, e.payload) for e in scenario.events
    ]


def test_scenario_generation_is_deterministic(toy_instance):
    spec = dict(
        instance=toy_instance,
        scenario_id="S1",
        horizon=120.0,
        new_order_ratio=0.4,
        breakdown_count=1,
        congestion_edge_ratio=0.2,
        congestion_multiplier=1.5,
    )
    a = generate_scenario(seed=123, **spec)
    b = generate_scenario(seed=123, **spec)
    c = generate_scenario(seed=124, **spec)
    assert _scenario_dict(a) == _scenario_dict(b)
    assert _scenario_dict(a) != _scenario_dict(c)


def test_scenario_seed_depends_on_run_and_spec_index():
    assert scenario_seed(2026, 11, 0) == scenario_seed(2026, 11, 0)
    assert scenario_seed(2026, 11, 0) != scenario_seed(2026, 12, 0)
    assert scenario_seed(2026, 11, 0) != scenario_seed(2026, 11, 1)


def _engine(instance, scenario, algorithm, seed):
    return SimulationEngine(
        instance=instance,
        scenario=scenario,
        algorithm=algorithm,
        seed=seed,
        planner_config=PlannerConfig(population_size=8, max_evaluations=160),
        sim_config=SimulationConfig(horizon=200.0),
    )


@pytest.mark.parametrize("algorithm", ["nsga2", "mopso", "moaco", "greedy"])
def test_same_seed_reproduces_metrics(toy_instance, algorithm):
    scenario = Scenario(scenario_id="S0", seed=1, events=[])
    r1 = _engine(toy_instance, scenario, algorithm, 17).run()
    r2 = _engine(toy_instance, scenario, algorithm, 17).run()
    # 运行时间指标由 wall clock 决定，不属于可复现性要求。
    timing_keys = {"replan_time_mean", "replan_time_max", "makespan"}
    m1 = {k: v for k, v in r1["metrics"].items() if k not in timing_keys}
    m2 = {k: v for k, v in r2["metrics"].items() if k not in timing_keys}
    assert m1 == m2
    assert [r["objectives"] for r in r1["replans"]] == [
        r["objectives"] for r in r2["replans"]
    ]


def test_algorithms_share_identical_event_schedule(toy_instance):
    scenario = generate_scenario(
        instance=toy_instance,
        scenario_id="S4",
        seed=999,
        horizon=150.0,
        new_order_ratio=0.2,
        breakdown_count=1,
        congestion_edge_ratio=0.1,
        congestion_multiplier=1.5,
    )
    results = [
        _engine(toy_instance, scenario, algo, 3).run()
        for algo in ("nsga2", "mopso", "greedy")
    ]
    schedules = [
        [(r["event_id"], r["event_time"]) for r in res["replans"][1:]]
        for res in results
    ]
    assert schedules[0] == schedules[1] == schedules[2]
    for res in results:
        # 5 个初始客户 + 1 个新增订单；故障后可能运力不足，但订单必须显式记账。
        assert res["metrics"]["released_orders"] == 6
        assert (
            res["metrics"]["served_orders"] + res["metrics"]["unserved_orders"] == 6
        )
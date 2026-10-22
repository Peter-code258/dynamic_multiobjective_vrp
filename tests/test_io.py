import json
import os

import pytest

from src.instance_loader import (
    InstanceFormatError,
    load_instance,
    load_json_instance,
    load_scenario,
    save_instance,
    save_scenario,
)
from src.models import DynamicEvent, Scenario


SOLOMON_SAMPLE = """3 100
CUSTOMER
0 50 50 0 0 1000 0
1 10 10 10 0 100 10
2 90 90 10 0 100 10
"""


def test_json_instance_roundtrip(tmp_path, toy_instance):
    path = str(tmp_path / "unit.json")
    save_instance(toy_instance, path)
    loaded = load_json_instance(path)
    assert loaded.instance_id == toy_instance.instance_id
    assert loaded.customer_ids == toy_instance.customer_ids
    assert loaded.vehicles[0].capacity == pytest.approx(60.0)
    assert loaded.customers[1].x == pytest.approx(20.0)


def test_solomon_parse_and_subset(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(SOLOMON_SAMPLE, encoding="utf-8")
    instance = load_instance(str(path))
    assert len(instance.customers) == 2
    assert instance.vehicles[0].capacity == pytest.approx(100.0)
    assert instance.customers[1].due_time == pytest.approx(100.0)
    assert instance.customers[1].release_time == pytest.approx(0.0)

    subset = load_instance_subset(str(path), max_customers=1, num_vehicles=2)
    assert len(subset.customers) == 1
    assert len(subset.vehicles) == 2
    assert subset.source == "solomon-derived"
    assert subset.metadata["derived_subset"] is True


def load_instance_subset(path, max_customers, num_vehicles):
    from src.instance_loader import load_solomon_instance

    return load_solomon_instance(
        path, instance_id="subset", max_customers=max_customers, num_vehicles=num_vehicles
    )


def test_missing_fields_raise(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"instance_id": "bad"}), encoding="utf-8")
    with pytest.raises(InstanceFormatError):
        load_json_instance(str(path))


def test_scenario_roundtrip(tmp_path):
    scenario = Scenario(
        scenario_id="S1",
        seed=5,
        events=[
            DynamicEvent("e1", "NEW_ORDER", 10.0, {"customer": {"customer_id": 1001}}),
            DynamicEvent("e2", "VEHICLE_BREAKDOWN", 20.0, {"vehicle_id": 0}),
        ],
        intensity={"new_order_ratio": 0.1},
    )
    path = str(tmp_path / "scenario.json")
    save_scenario(scenario, path)
    loaded = load_scenario(path)
    assert loaded.scenario_id == "S1"
    assert [e.event_type for e in loaded.events] == ["NEW_ORDER", "VEHICLE_BREAKDOWN"]
    assert loaded.events[0].payload["customer"]["customer_id"] == 1001


def test_scenario_missing_event_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"scenario_id": "x", "seed": 1, "events": [{"time": 1}]}), encoding="utf-8")
    with pytest.raises(InstanceFormatError):
        load_scenario(str(path))

import os

import pytest

from src.context import PlanningContext
from src.models import Customer, VRPInstance, Vehicle
from src.travel_time import TravelNetwork


@pytest.fixture()
def toy_instance() -> VRPInstance:
    customers = {
        1: Customer(customer_id=1, x=20.0, y=0.0, demand=20.0, service_time=1.0),
        2: Customer(customer_id=2, x=40.0, y=0.0, demand=20.0, service_time=1.0),
        3: Customer(customer_id=3, x=20.0, y=30.0, demand=20.0, service_time=1.0),
        4: Customer(customer_id=4, x=40.0, y=30.0, demand=20.0, service_time=1.0),
        5: Customer(customer_id=5, x=60.0, y=15.0, demand=20.0, service_time=1.0),
    }
    vehicles = {
        0: Vehicle(vehicle_id=0, capacity=60.0),
        1: Vehicle(vehicle_id=1, capacity=60.0),
    }
    return VRPInstance(
        instance_id="unit",
        depot_id=0,
        depot_x=0.0,
        depot_y=0.0,
        vehicles=vehicles,
        customers=customers,
        source="unit-test",
    )


@pytest.fixture()
def travel(toy_instance: VRPInstance) -> TravelNetwork:
    return TravelNetwork(toy_instance, speed=1.0)


@pytest.fixture()
def context(toy_instance: VRPInstance, travel: TravelNetwork) -> PlanningContext:
    return PlanningContext.from_state(
        instance=toy_instance,
        travel=travel,
        remaining=list(toy_instance.customers.keys()),
        used_vehicle_ids=set(),
    )


@pytest.fixture()
def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

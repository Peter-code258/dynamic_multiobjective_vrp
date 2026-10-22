import pytest

from src.travel_time import TravelNetwork


def test_distance_symmetric_and_zero(toy_instance):
    net = TravelNetwork(toy_instance)
    assert net.distance(0, 1) == pytest.approx(20.0)
    assert net.distance(1, 0) == pytest.approx(20.0)
    assert net.distance(2, 2) == 0.0


def test_congestion_multiplier_depends_on_departure_time(toy_instance):
    net = TravelNetwork(toy_instance)
    net.add_congestion("c1", start=100.0, duration=50.0, edges=[(0, 1)], multiplier=2.0)
    assert net.travel_time(0, 1, 0.0) == pytest.approx(20.0)
    assert net.travel_time(0, 1, 120.0) == pytest.approx(40.0)
    assert net.travel_time(0, 1, 160.0) == pytest.approx(20.0)
    # 距离不因拥堵改变
    assert net.distance(0, 1) == pytest.approx(20.0)


def test_congestion_uses_max_multiplier_when_overlapping(toy_instance):
    net = TravelNetwork(toy_instance)
    net.add_congestion("c1", start=0.0, duration=100.0, edges=[(0, 1)], multiplier=1.5)
    net.add_congestion("c2", start=10.0, duration=50.0, edges=[(1, 0)], multiplier=2.0)
    assert net.active_multiplier(0, 1, 20.0) == pytest.approx(2.0)
    net.remove_congestion("c2")
    assert net.active_multiplier(0, 1, 20.0) == pytest.approx(1.5)


def test_invalid_speed_and_multiplier(toy_instance):
    with pytest.raises(ValueError):
        TravelNetwork(toy_instance, speed=0)
    net = TravelNetwork(toy_instance)
    with pytest.raises(ValueError):
        net.add_congestion("bad", 0.0, 10.0, [(0, 1)], multiplier=0.5)

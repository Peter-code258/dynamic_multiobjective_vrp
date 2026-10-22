import math

import pytest

from src.metrics import (
    hypervolume,
    igd,
    igd_plus,
    normalize_points,
    shared_bounds,
)
from src.statistics import (
    descriptive_stats,
    friedman_test,
    holm_correction,
    paired_wilcoxon,
)


def test_hypervolume_two_dimensional_exact():
    points = [(0.2, 0.6), (0.6, 0.2)]
    assert hypervolume(points, (1.0, 1.0)) == pytest.approx(0.48)


def test_hypervolume_single_point_and_empty():
    assert hypervolume([(0.5, 0.5)], (1.0, 1.0)) == pytest.approx(0.25)
    assert hypervolume([], (1.0, 1.0)) == 0.0
    assert hypervolume([(1.5, 0.5)], (1.0, 1.0)) == 0.0


def test_hypervolume_4d_monotone():
    ref = (1.1, 1.1, 1.1, 1.1)
    a = [(0.1, 0.5, 0.5, 0.5)]
    b = [(0.3, 0.5, 0.5, 0.5)]
    assert hypervolume(a, ref) > hypervolume(b, ref)


def test_igda_and_igd_plus_zero_on_identical_front():
    points = [(0.1, 0.2, 0.3, 0.4), (0.5, 0.4, 0.2, 0.1)]
    assert igd(points, points) == pytest.approx(0.0)
    assert igd_plus(points, points) == pytest.approx(0.0)
    assert igd_plus(points, [(0.0, 0.0, 0.0, 0.0)]) == pytest.approx(0.0)


def test_igda_nan_on_empty():
    assert math.isnan(igd([(0.1, 0.2)], []))


def test_normalization_shared_bounds():
    points = [[0.0, 10.0], [10.0, 20.0]]
    bounds = shared_bounds(points, pad=0.0)
    normalized = normalize_points(points, bounds)
    assert normalized[0] == pytest.approx([0.0, 0.0])
    assert normalized[1] == pytest.approx([1.0, 1.0])


def test_descriptive_stats():
    stats = descriptive_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]


def test_wilcoxon_identical_arrays():
    result = paired_wilcoxon([1, 2, 3, 4], [1, 2, 3, 4])
    assert result["p_value"] == pytest.approx(1.0)
    assert result["effect_rank_biserial"] == pytest.approx(0.0)


def test_wilcoxon_clear_difference():
    result = paired_wilcoxon([10, 11, 12, 13, 14], [1, 2, 3, 4, 5])
    assert result["p_value"] < 0.1
    assert result["effect_rank_biserial"] == pytest.approx(1.0)


def test_holm_correction_monotone():
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted == sorted(adjusted)


def test_friedman_requires_blocks():
    result = friedman_test([[1, 2], [2, 3]])
    assert math.isnan(result["p_value"])

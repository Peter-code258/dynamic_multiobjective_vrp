import pytest

from src.models import Solution
from src.pareto import (
    ParetoArchive,
    constrained_dominates,
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
)


def _sol(objectives, feasible=True, violation=0.0):
    return Solution(routes={0: [1]}, objectives=tuple(objectives), feasible=feasible, violation_score=violation)


def test_dominance_definition():
    assert dominates([1, 2], [2, 2])
    assert dominates([1, 2], [1, 3])
    assert not dominates([1, 2], [1, 2])
    assert not dominates([2, 1], [1, 2])


def test_constrained_dominance_prefers_feasible():
    assert constrained_dominates(_sol([10, 10], True), _sol([1, 1], False, 1.0))
    assert constrained_dominates(_sol([1, 1], False, 1.0), _sol([1, 1], False, 2.0))


def test_fast_non_dominated_sort_layers():
    sols = [_sol([1, 4]), _sol([2, 2]), _sol([4, 1]), _sol([3, 3])]
    fronts = fast_non_dominated_sort(sols)
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]


def test_crowding_distance_boundaries_infinite():
    sols = [_sol([1, 4]), _sol([2, 2]), _sol([4, 1])]
    cd = crowding_distance(sols, [0, 1, 2])
    assert cd[0] == float("inf")
    assert cd[2] == float("inf")
    assert cd[1] > 0


def test_archive_keeps_non_dominated_only():
    archive = ParetoArchive(max_size=10)
    assert archive.add(_sol([1, 4]))
    assert archive.add(_sol([4, 1]))
    assert archive.add(_sol([3, 3]))
    assert not archive.add(_sol([4, 4]))
    assert len(archive) == 3


def test_archive_rejects_infeasible_by_default():
    archive = ParetoArchive(max_size=10)
    assert not archive.add(_sol([1, 1], feasible=False, violation=5.0))
    assert len(archive) == 0
    archive2 = ParetoArchive(max_size=10, feasible_only=False)
    assert archive2.add(_sol([1, 1], feasible=False, violation=5.0))


def test_archive_prunes_to_max_size_and_keeps_extremes():
    archive = ParetoArchive(max_size=4)
    points = [[0, 10], [2, 8], [4, 6], [6, 4], [8, 2], [10, 0]]
    for p in points:
        archive.add(_sol(p))
    assert len(archive) == 4
    vectors = {tuple(s.objectives) for s in archive.solutions}
    assert (0, 10) in vectors and (10, 0) in vectors

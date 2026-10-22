"""多目标算法注册表。"""
from __future__ import annotations

from typing import Callable, Dict

from .base import AlgorithmResult, PlannerConfig
from .moaco import run_moaco
from .mopso import run_mopso
from .nsga2 import run_nsga2


PLANNERS: Dict[str, Callable] = {
    "nsga2": run_nsga2,
    "mopso": run_mopso,
    "moaco": run_moaco,
}


def get_planner(name: str) -> Callable:
    key = name.strip().lower()
    if key not in PLANNERS:
        raise ValueError("未知算法: %s（可用: %s）" % (name, ", ".join(PLANNERS)))
    return PLANNERS[key]

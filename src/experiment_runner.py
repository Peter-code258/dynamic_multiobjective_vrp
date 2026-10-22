"""统一实验运行器：数据准备、S0-S4 运行、原始结果、汇总、统计与图表。"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .algorithms.base import PlannerConfig
from .instance_loader import load_instance, load_scenario, save_scenario
from .metrics import hypervolume, igd, igd_plus, normalize_points, shared_bounds
from .pareto import Solution, fast_non_dominated_sort
from .simulator import SimulationConfig, SimulationEngine
from .statistics import (
    descriptive_table,
    friedman_table,
    pairwise_wilcoxon_table,
)
from .visualization import (
    plot_metric_boxplot,
    plot_metric_lines,
    plot_pareto_projections,
    plot_route_comparison,
    plot_route_map,
)
from .scenario_generator import (
    generate_scenario,
    generate_scenario_suite,
    generate_toy_instances,
    scenario_seed,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARETO_ALGORITHMS = ("nsga2", "mopso", "moaco")


def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_config(path: str) -> Dict[str, object]:
    with open(resolve(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件必须是 YAML 映射")
    return cfg


def config_hash(cfg: Dict[str, object]) -> str:
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _instance_specs(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    specs = cfg.get("instance_specs", [])
    if not specs:
        raise ValueError("配置缺少 instance_specs")
    return list(specs)


def _instance_paths(cfg: Dict[str, object]) -> Dict[str, str]:
    """优先使用显式实例文件（例如 Solomon），否则生成小规模合成实例。"""
    explicit = cfg.get("instance_files", [])
    if explicit:
        paths = [resolve(str(path)) for path in explicit]
        return {load_instance(path).instance_id: path for path in paths}
    instance_dir = resolve(str(cfg.get("paths", {}).get("instance_dir", "data/instances")))
    specs = _instance_specs(cfg)
    paths = generate_toy_instances(instance_dir, specs)
    return {str(spec["instance_id"]): path for spec, path in zip(specs, paths)}


def _scenario_horizon(cfg: Dict[str, object]) -> Optional[float]:
    """配置 horizon 为 null/0 时按实例自动估计运营周期。"""
    raw = cfg.get("scenarios", {}).get("horizon", None)
    if raw is None:
        return None
    value = float(raw)
    return None if value <= 0 else value


def _scenario_specs(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    return list(cfg.get("scenarios", {}).get("specs", [{"id": "S0"}]))


def _scenario_path(scenario_dir: str, instance_id: str, scenario_id: str, seed: int) -> str:
    return os.path.join(scenario_dir, "%s__%s__s%d.json" % (instance_id, scenario_id, seed))


def prepare_data(
    cfg: Dict[str, object], seeds: Optional[Sequence[int]] = None
) -> Tuple[Dict[str, str], List[str]]:
    scenario_dir = resolve(str(cfg.get("paths", {}).get("scenario_dir", "data/scenarios")))
    instance_paths = _instance_paths(cfg)

    scenario_cfg = cfg.get("scenarios", {})
    horizon = _scenario_horizon(cfg)
    base_seed = int(scenario_cfg.get("base_seed", 2026))
    selected_seeds = [int(s) for s in (seeds or cfg.get("experiment", {}).get("seeds", [0]))]
    speed = float(cfg.get("simulation", {}).get("speed", 1.0))
    scenario_paths: List[str] = []
    for iid, path in instance_paths.items():
        instance = load_instance(path)
        generated = generate_scenario_suite(
            instance=instance,
            output_dir=scenario_dir,
            specs=_scenario_specs(cfg),
            base_seed=base_seed,
            horizon=horizon,
            seeds=selected_seeds,
            speed=speed,
        )
        scenario_paths.extend(generated)
    return instance_paths, scenario_paths


def _planner_config(
    cfg: Dict[str, object],
    max_evaluations: Optional[int],
    overrides: Optional[Dict[str, object]] = None,
) -> PlannerConfig:
    base = cfg.get("planner", {})
    overrides = overrides or {}
    extra = {
        "c1": float(base.get("c1", 0.6)),
        "c2": float(base.get("c2", 0.6)),
        "inertia_strength": float(base.get("inertia_strength", 2)),
        "alpha": float(base.get("alpha", 1.0)),
        "beta": float(base.get("beta", 1.0)),
    }
    population_size = int(overrides.get("population_size", base.get("population_size", 50)))
    archive_size = int(overrides.get("archive_size", base.get("archive_size", 100)))
    response_strategy = str(
        overrides.get("response_strategy", base.get("response_strategy", "warm_start"))
    )
    configured_budget = overrides.get("max_evaluations", base.get("max_evaluations", 5000))
    effective_budget = max_evaluations if max_evaluations is not None else configured_budget
    return PlannerConfig(
        population_size=population_size,
        max_evaluations=int(effective_budget),
        archive_size=archive_size,
        crossover_rate=float(base.get("crossover_rate", 0.9)),
        mutation_rate=float(base.get("mutation_rate", 0.15)),
        response_strategy=response_strategy,
        pheromone_evaporation=float(base.get("pheromone_evaporation", 0.2)),
        extra=extra,
    )


def _planner_variants(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    """返回待运行的算法参数组合。

    显式 `planner.variants` 优先；否则由 population_sizes / evaluation_budgets /
    response_strategies 做笛卡尔积，未提供时回退到单组默认参数。
    """
    base = cfg.get("planner", {})
    explicit = base.get("variants")
    variants: List[Dict[str, object]] = []
    if explicit:
        for idx, item in enumerate(explicit):
            variants.append(
                {
                    "variant_id": str(item.get("id", "v%d" % idx)),
                    "population_size": int(
                        item.get("population_size", base.get("population_size", 50))
                    ),
                    "max_evaluations": int(
                        item.get("max_evaluations", base.get("max_evaluations", 5000))
                    ),
                    "archive_size": int(
                        item.get("archive_size", base.get("archive_size", 100))
                    ),
                    "response_strategy": str(
                        item.get("response_strategy", base.get("response_strategy", "warm_start"))
                    ),
                }
            )
        return variants
    populations = base.get("population_sizes") or [base.get("population_size", 50)]
    budgets = base.get("evaluation_budgets") or [base.get("max_evaluations", 5000)]
    strategies = base.get("response_strategies") or [
        base.get("response_strategy", "warm_start")
    ]
    for pop in populations:
        for budget in budgets:
            for strategy in strategies:
                variants.append(
                    {
                        "variant_id": "p%s_b%s_%s" % (pop, budget, strategy),
                        "population_size": int(pop),
                        "max_evaluations": int(budget),
                        "archive_size": int(base.get("archive_size", 100)),
                        "response_strategy": str(strategy),
                    }
                )
    return variants


def run_single(
    instance_path: str,
    scenario_path: str,
    algorithm: str,
    seed: int,
    cfg: Dict[str, object],
    output_dir: str,
    max_evaluations: Optional[int] = None,
    planner_overrides: Optional[Dict[str, object]] = None,
    variant_id: Optional[str] = None,
) -> Dict[str, object]:
    instance = load_instance(instance_path)
    scenario = load_scenario(scenario_path)
    configured_horizon = _scenario_horizon(cfg)
    sim_horizon = (
        configured_horizon
        if configured_horizon is not None
        else float(scenario.metadata.get("horizon", 600.0))
    )
    sim_cfg = SimulationConfig(
        horizon=sim_horizon,
        speed=float(cfg.get("simulation", {}).get("speed", 1.0)),
        max_route_duration=cfg.get("simulation", {}).get("max_route_duration"),
        hard_time_windows=bool(cfg.get("simulation", {}).get("hard_time_windows", False)),
        selection_rule=str(cfg.get("simulation", {}).get("selection_rule", "ideal_point")),
    )
    planner_cfg = _planner_config(cfg, max_evaluations, planner_overrides)
    engine = SimulationEngine(
        instance=instance,
        scenario=scenario,
        algorithm=algorithm,
        seed=int(seed),
        planner_config=planner_cfg,
        sim_config=sim_cfg,
    )
    result = engine.run()
    result["instance"] = {
        "depot_id": instance.depot_id,
        "depot_x": instance.depot_x,
        "depot_y": instance.depot_y,
        "source": instance.source,
        "customer_count": len(instance.customers),
        "vehicle_count": len(instance.vehicles),
    }
    result["scenario_intensity"] = dict(scenario.intensity)
    result["scenario_metadata"] = dict(scenario.metadata)
    result["planner_config"] = {
        **planner_cfg.__dict__,
        **result.get("planner_config", {}),
    }
    result["variant_id"] = variant_id or "default"
    result["run_id"] = "%s__%s__%s__s%s__%s__e%d" % (
        instance.instance_id,
        scenario.scenario_id,
        algorithm,
        seed,
        result["variant_id"],
        planner_cfg.max_evaluations,
    )
    result["instance_path"] = instance_path
    result["scenario_path"] = scenario_path
    result["config_hash"] = config_hash(cfg)
    result["created_at"] = datetime.now().isoformat(timespec="seconds")
    result["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit": _software_info()["git_commit"],
    }

    run_dir = os.path.join(output_dir, "raw")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, result["run_id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _ensure_scenario(
    cfg: Dict[str, object], instance, scenario_dir: str, scenario_id: str, seed: int
) -> Optional[str]:
    """返回 (实例, 场景, 种子) 对应的事件文件；缺失时按同一随机流规则即时生成。"""
    path = _scenario_path(scenario_dir, instance.instance_id, scenario_id, seed)
    if os.path.exists(path):
        return path
    specs = _scenario_specs(cfg)
    spec_index = {str(spec["id"]): idx for idx, spec in enumerate(specs)}
    if scenario_id not in spec_index:
        return None
    idx = spec_index[scenario_id]
    spec = specs[idx]
    scenario = generate_scenario(
        instance=instance,
        scenario_id=scenario_id,
        seed=scenario_seed(int(cfg.get("scenarios", {}).get("base_seed", 2026)), seed, idx),
        horizon=_scenario_horizon(cfg),
        new_order_ratio=float(spec.get("new_order_ratio", 0.0)),
        breakdown_count=int(spec.get("breakdown_count", 0)),
        breakdown_ratio=float(spec.get("breakdown_ratio", 0.0)),
        congestion_edge_ratio=float(spec.get("congestion_edge_ratio", 0.0)),
        congestion_multiplier=float(spec.get("congestion_multiplier", 1.0)),
        congestion_duration_ratio=float(spec.get("congestion_duration_ratio", 0.25)),
        simultaneous=bool(spec.get("simultaneous", False)),
        speed=float(cfg.get("simulation", {}).get("speed", 1.0)),
    )
    save_scenario(scenario, path)
    return path


def run_experiment(
    cfg: Dict[str, object],
    instances: Optional[Sequence[str]] = None,
    scenarios: Optional[Sequence[str]] = None,
    algorithms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    max_evaluations: Optional[int] = None,
    output_dir: Optional[str] = None,
    prepare: bool = True,
) -> pd.DataFrame:
    if prepare:
        instance_paths, _ = prepare_data(cfg)
    else:
        explicit = cfg.get("instance_files", [])
        if explicit:
            instance_paths = {
                load_instance(resolve(str(path))).instance_id: resolve(str(path))
                for path in explicit
            }
        else:
            instance_dir = resolve(str(cfg.get("paths", {}).get("instance_dir", "data/instances")))
            instance_paths = {
                str(spec["instance_id"]): os.path.join(instance_dir, "%s.json" % spec["instance_id"])
                for spec in _instance_specs(cfg)
            }
    scenario_dir = resolve(str(cfg.get("paths", {}).get("scenario_dir", "data/scenarios")))

    exp = cfg.get("experiment", {})
    selected_instances = list(instances or exp.get("instances", list(instance_paths.keys())))
    selected_scenarios = list(scenarios or exp.get("scenarios", ["S0"]))
    selected_algorithms = list(algorithms or cfg.get("planner", {}).get("algorithms", ["nsga2"]))
    selected_seeds = [int(s) for s in (seeds or exp.get("seeds", [11, 12, 13]))]
    out_dir = resolve(str(output_dir or exp.get("output_dir", "results/default")))
    os.makedirs(out_dir, exist_ok=True)
    if prepare:
        prepare_data(cfg, seeds=selected_seeds)

    variants = _planner_variants(cfg)
    total = (
        len(selected_instances) * len(selected_scenarios)
        * len(selected_seeds) * len(selected_algorithms) * len(variants)
    )
    done = 0
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for iid in selected_instances:
        if iid not in instance_paths:
            print("[skip] 未找到实例 %s" % iid, flush=True)
            continue
        instance = load_instance(instance_paths[iid])
        for sid in selected_scenarios:
            for seed in selected_seeds:
                scenario_path = _ensure_scenario(cfg, instance, scenario_dir, sid, seed)
                if scenario_path is None or not os.path.exists(scenario_path):
                    print("[skip] 未找到场景 %s / %s / seed=%s" % (iid, sid, seed), flush=True)
                    continue
                for variant in variants:
                    for algo in selected_algorithms:
                        done += 1
                        started = time.perf_counter()
                        try:
                            result = run_single(
                                instance_paths[iid],
                                scenario_path,
                                algo,
                                seed,
                                cfg,
                                out_dir,
                                max_evaluations=max_evaluations,
                                planner_overrides=variant,
                                variant_id=str(variant["variant_id"]),
                            )
                        except Exception as exc:  # 失败运行必须留痕，不能静默丢弃
                            failures.append(
                                {
                                    "instance_id": iid,
                                    "scenario_id": sid,
                                    "seed": seed,
                                    "algorithm": algo,
                                    "variant_id": variant["variant_id"],
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            )
                            print(
                                "[%d/%d] FAIL %s %s %s seed=%s variant=%s: %s"
                                % (done, total, iid, sid, algo, seed, variant["variant_id"], exc),
                                flush=True,
                            )
                            continue
                        metrics = result["metrics"]
                        row = {
                            "run_id": result["run_id"],
                            "instance_id": iid,
                            "scenario_id": sid,
                            "seed": seed,
                            "algorithm": algo,
                            "variant_id": result["variant_id"],
                            "population_size": int(result["planner_config"]["population_size"]),
                            "max_evaluations": int(result["planner_config"]["max_evaluations"]),
                            "response_strategy": result["planner_config"]["response_strategy"],
                            "new_order_ratio": result["scenario_intensity"].get("new_order_ratio", 0.0),
                            "breakdown_count": result["scenario_intensity"].get("breakdown_count", 0),
                            "congestion_edge_ratio": result["scenario_intensity"].get("congestion_edge_ratio", 0.0),
                            "congestion_multiplier": result["scenario_intensity"].get("congestion_multiplier", 1.0),
                            "archive_size": len(result["archive"]),
                            "wall_time_s": result["wall_time_s"],
                            "run_wall_s": time.perf_counter() - started,
                        }
                        row.update(metrics)
                        rows.append(row)
                        print(
                            "[%d/%d] %s %s %s seed=%s variant=%s feasible=%s service=%.3f f1=%.2f f3=%.2f eval=%s"
                            % (
                                done, total, iid, sid, algo, seed, result["variant_id"],
                                metrics["feasible"], metrics["service_rate"],
                                metrics["f1_total_distance"], metrics["f3_total_response_time"],
                                metrics["evaluations_total"],
                            ),
                            flush=True,
                        )

    if failures:
        with open(os.path.join(out_dir, "failures.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame(rows)
    if not df.empty:
        os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
        df.to_csv(os.path.join(out_dir, "summary", "runs.csv"), index=False, encoding="utf-8-sig")
    return df


def _sha256_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _software_info() -> Dict[str, object]:
    from importlib import metadata

    versions: Dict[str, str] = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib", "PyYAML"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    git_commit = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": git_commit,
    }


def analyze_results(
    output_dir: str,
    cfg: Optional[Dict[str, object]] = None,
    make_figures: bool = True,
) -> Dict[str, pd.DataFrame]:
    out_dir = resolve(output_dir)
    raw_dir = os.path.join(out_dir, "raw")
    summary_dir = os.path.join(out_dir, "summary")
    figure_dir = os.path.join(out_dir, "figures")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []
    archives: Dict[Tuple[str, str], Dict[str, List[List[float]]]] = {}
    group_points: Dict[Tuple[str, str], List[List[float]]] = {}
    run_points: Dict[str, List[List[float]]] = {}
    route_payloads: Dict[str, Dict[str, object]] = {}

    for name in sorted(os.listdir(raw_dir)) if os.path.isdir(raw_dir) else []:
        if not name.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, name), "r", encoding="utf-8") as f:
            result = json.load(f)
        run = result["run"]
        iid, sid, algo, seed = (
            run["instance_id"], run["scenario_id"], run["algorithm"], run["seed"]
        )
        metrics = result["metrics"]
        run_id = result.get("run_id", name[:-5])
        row = {
            "run_id": run_id,
            "instance_id": iid,
            "scenario_id": sid,
            "seed": seed,
            "algorithm": algo,
            "variant_id": result.get("variant_id", "default"),
            "population_size": result.get("planner_config", {}).get("population_size"),
            "max_evaluations": result.get("planner_config", {}).get("max_evaluations"),
            "response_strategy": result.get("planner_config", {}).get("response_strategy"),
            "archive_size": len(result.get("archive", [])),
            "wall_time_s": result.get("wall_time_s", float("nan")),
            "new_order_ratio": result.get("scenario_intensity", {}).get("new_order_ratio", 0.0),
            "breakdown_count": result.get("scenario_intensity", {}).get("breakdown_count", 0),
            "congestion_edge_ratio": result.get("scenario_intensity", {}).get("congestion_edge_ratio", 0.0),
            "congestion_multiplier": result.get("scenario_intensity", {}).get("congestion_multiplier", 1.0),
        }
        row.update(metrics)
        rows.append(row)
        points = [
            [float(x) for x in item["objectives"]]
            for item in result.get("archive", [])
            if item.get("feasible", True)
        ]
        archives.setdefault((iid, sid), {}).setdefault(algo, []).extend(points)
        if algo in PARETO_ALGORITHMS:
            # 经验参考前沿只由三种待比较算法的所有重复运行合并构造（md 11.3）；
            # 贪心基线是工程参照，不参与 HV/IGD 参考集与算法间统计。
            group_points.setdefault((iid, sid), []).extend(points)
        run_points[run_id] = points
        route_payloads.setdefault(algo, {})[run_id] = result

    df = pd.DataFrame(rows)
    if df.empty:
        return {"runs": df}

    # 运行级 HV/IGD：同一（实例,场景）内共享归一化上下界与经验参考前沿，
    # 参考集来自三种算法所有种子的最终档案并集。
    group_cache: Dict[Tuple[str, str], Tuple[List, List]] = {}
    for (iid, sid), all_points in group_points.items():
        if not all_points:
            continue
        bounds = shared_bounds(all_points)
        dummy = [Solution(routes={}, objectives=tuple(p), feasible=True) for p in all_points]
        fronts = fast_non_dominated_sort(dummy)
        ref_front = [all_points[i] for i in fronts[0]] if fronts else all_points
        group_cache[(iid, sid)] = (bounds, ref_front)

    run_quality: Dict[str, Dict[str, float]] = {}
    for run_id, points in run_points.items():
        row = next(r for r in rows if r["run_id"] == run_id)
        if str(row["algorithm"]) not in PARETO_ALGORITHMS:
            run_quality[run_id] = {"hv": float("nan"), "igd": float("nan"), "igd_plus": float("nan")}
            continue
        key = (str(row["instance_id"]), str(row["scenario_id"]))
        if not points or key not in group_cache:
            run_quality[run_id] = {"hv": 0.0, "igd": float("nan"), "igd_plus": float("nan")}
            continue
        bounds, ref_front = group_cache[key]
        norm_points = normalize_points(points, bounds)
        norm_ref = normalize_points(ref_front, bounds)
        run_quality[run_id] = {
            "hv": hypervolume(norm_points, (1.1, 1.1, 1.1, 1.1)),
            "igd": igd(norm_ref, norm_points),
            "igd_plus": igd_plus(norm_ref, norm_points),
        }
    df["hv"] = df["run_id"].map(lambda r: run_quality.get(str(r), {}).get("hv", float("nan")))
    df["igd"] = df["run_id"].map(lambda r: run_quality.get(str(r), {}).get("igd", float("nan")))
    df["igd_plus"] = df["run_id"].map(lambda r: run_quality.get(str(r), {}).get("igd_plus", float("nan")))
    df.to_csv(os.path.join(summary_dir, "runs.csv"), index=False, encoding="utf-8-sig")

    # 车辆任务表：每行一辆车在某次运行中的任务、载荷与距离（md 工程意义要求）。
    task_rows: List[Dict[str, object]] = []
    for algo, payloads in route_payloads.items():
        for run_id, result in payloads.items():
            run_meta = result.get("run", {})
            customers = result.get("customers", {})
            for vid, vehicle in result.get("vehicles", {}).items():
                planned = list(result.get("final_plan", {}).get(vid, []))
                planned_load = sum(
                    float(customers.get(str(cid), {}).get("demand", 0.0)) for cid in planned
                )
                capacity = float(vehicle.get("capacity", 0.0))
                task_rows.append(
                    {
                        "run_id": run_id,
                        "instance_id": run_meta.get("instance_id"),
                        "scenario_id": run_meta.get("scenario_id"),
                        "seed": run_meta.get("seed"),
                        "algorithm": algo,
                        "variant_id": result.get("variant_id"),
                        "vehicle_id": int(vid),
                        "status": vehicle.get("status"),
                        "used": vehicle.get("used"),
                        "capacity": capacity,
                        "served_customers": " ".join(
                            str(c) for c in vehicle.get("served_customers", [])
                        ),
                        "served_demand": vehicle.get("cumulative_demand"),
                        "served_distance": vehicle.get("cumulative_distance"),
                        "utilization": (
                            float(vehicle.get("cumulative_demand", 0.0)) / capacity
                            if capacity > 0 else None
                        ),
                        "last_planned_route": " ".join(str(c) for c in planned),
                        "last_planned_load": planned_load,
                    }
                )
    pd.DataFrame(task_rows).to_csv(
        os.path.join(summary_dir, "vehicle_tasks.csv"), index=False, encoding="utf-8-sig"
    )

    metrics = [
        "f1_total_distance", "f2_used_vehicles", "f3_total_response_time",
        "f4_load_utilization_variance", "service_rate", "replan_time_mean",
        "route_change_rate_mean", "recovery_rate", "recovery_time_mean",
        "unrecovered_events", "hv", "igd", "igd_plus",
    ]
    desc = descriptive_table(df, ["instance_id", "scenario_id", "algorithm"], metrics)
    desc.to_csv(os.path.join(summary_dir, "descriptive_by_scenario.csv"), index=False, encoding="utf-8-sig")
    overall = descriptive_table(df, ["algorithm"], metrics)
    overall.to_csv(os.path.join(summary_dir, "descriptive_overall.csv"), index=False, encoding="utf-8-sig")

    pair_frames = []
    for metric in [
        "f1_total_distance", "f2_used_vehicles", "f3_total_response_time",
        "f4_load_utilization_variance", "service_rate", "replan_time_mean",
        "route_change_rate_mean", "recovery_time_mean", "hv", "igd",
    ]:
        table = pairwise_wilcoxon_table(df, metric, list(PARETO_ALGORITHMS))
        if not table.empty:
            pair_frames.append(table)
    if pair_frames:
        pd.concat(pair_frames).to_csv(
            os.path.join(summary_dir, "wilcoxon_pairwise.csv"), index=False, encoding="utf-8-sig"
        )
    friedman_frames = []
    for metric in [
        "f1_total_distance", "f2_used_vehicles", "f3_total_response_time",
        "f4_load_utilization_variance", "service_rate", "replan_time_mean",
        "hv", "igd",
    ]:
        table = friedman_table(df, metric, list(PARETO_ALGORITHMS))
        if not table.empty:
            friedman_frames.append(table)
    if friedman_frames:
        pd.concat(friedman_frames).to_csv(
            os.path.join(summary_dir, "friedman_tests.csv"), index=False, encoding="utf-8-sig"
        )

    if make_figures:
        for metric in [
            "f1_total_distance", "f2_used_vehicles", "f3_total_response_time",
            "f4_load_utilization_variance", "service_rate", "replan_time_mean", "hv",
        ]:
            plot_metric_boxplot(
                df, metric,
                os.path.join(figure_dir, "box_%s.png" % metric),
                title=metric,
            )
        # 新增订单强度与拥堵强度趋势
        s1 = df[df["scenario_id"].astype(str).str.startswith("S1")]
        if not s1.empty:
            plot_metric_lines(
                s1, "new_order_ratio", "service_rate", "algorithm",
                os.path.join(figure_dir, "line_new_order_service_rate.png"),
                title="New-order intensity vs service rate",
            )
            plot_metric_lines(
                s1, "new_order_ratio", "f3_total_response_time", "algorithm",
                os.path.join(figure_dir, "line_new_order_response.png"),
                title="New-order intensity vs total response time",
            )
        s3 = df[df["scenario_id"].astype(str).str.startswith("S3")]
        if not s3.empty:
            plot_metric_lines(
                s3, "congestion_multiplier", "f3_total_response_time", "algorithm",
                os.path.join(figure_dir, "line_congestion_response.png"),
                title="Congestion multiplier vs total response time",
            )
            plot_metric_lines(
                s3, "congestion_multiplier", "replan_time_mean", "algorithm",
                os.path.join(figure_dir, "line_congestion_replan_time.png"),
                title="Congestion multiplier vs replanning time",
            )
        # Pareto 投影：每个实例的 S0 与 S4
        for (iid, sid), by_algo in archives.items():
            if sid not in ("S0", "S4"):
                continue
            plot_pareto_projections(
                by_algo,
                os.path.join(figure_dir, "pareto_%s_%s.png" % (iid, sid)),
                title="Pareto projections: %s / %s" % (iid, sid),
            )
        # 路线图：每个算法选一张车辆故障场景
        for algo, payloads in route_payloads.items():
            candidates = [
                r for r in payloads.values()
                if r["run"]["scenario_id"] in ("S2", "S4")
            ]
            if not candidates:
                continue
            chosen = max(
                candidates,
                key=lambda r: sum(len(x) for x in r.get("final_plan", {}).values()),
            )
            inst = chosen["instance"]
            plot_route_map(
                chosen["customers"],
                chosen["final_plan"],
                os.path.join(figure_dir, "route_%s_%s.png" % (algo, chosen["run"]["scenario_id"])),
                depot=(inst["depot_x"], inst["depot_y"]),
                title="%s / %s / %s / seed %s" % (
                    algo, chosen["run"]["instance_id"],
                    chosen["run"]["scenario_id"], chosen["run"]["seed"]
                ),
                vehicles=chosen.get("vehicles"),
            )
            # 故障前后路线对比（md 12.5）：选受影响客户最多、重规划后仍有任务的运行
            best_break: Tuple[int, object, int] = None  # type: ignore[assignment]
            for result in payloads.values():
                replans = result.get("replans", [])
                for i, item in enumerate(replans):
                    if item.get("event_type") != "VEHICLE_BREAKDOWN" or i == 0:
                        continue
                    after = item.get("plan", {})
                    score = (
                        len(item.get("affected_customers", [])),
                        sum(len(route) for route in after.values()),
                    )
                    if best_break is None or score > best_break[0]:
                        best_break = (score, result, i)  # type: ignore[assignment]
            if best_break is not None:
                _, result, index = best_break
                replans = result["replans"]
                before = replans[index - 1].get("plan", {})
                after = replans[index].get("plan", {})
                inst = result["instance"]
                plot_route_comparison(
                    result["customers"],
                    before,
                    after,
                    os.path.join(
                        figure_dir,
                        "route_breakdown_%s_%s.png"
                        % (algo, result["run"]["instance_id"]),
                    ),
                    depot=(inst["depot_x"], inst["depot_y"]),
                    title="%s / %s / %s / seed %s" % (
                        algo, result["run"]["instance_id"],
                        result["run"]["scenario_id"], result["run"]["seed"],
                    ),
                    affected=replans[index].get("affected_customers", []),
                )

    data_hashes: Dict[str, str] = {}
    for payload in route_payloads.values():
        for result in payload.values():
            for key in ("instance_path", "scenario_path"):
                path = result.get(key)
                if path and os.path.exists(str(path)):
                    digest = _sha256_file(str(path))
                    if digest:
                        data_hashes[os.path.basename(str(path))] = digest
    manifest = {
        "output_dir": out_dir,
        "runs": int(len(df)),
        "instances": sorted(df["instance_id"].unique().tolist()),
        "scenarios": sorted(df["scenario_id"].unique().tolist()),
        "algorithms": sorted(df["algorithm"].unique().tolist()),
        "seeds": sorted(int(s) for s in df["seed"].unique().tolist()),
        "variants": sorted(str(v) for v in df["variant_id"].unique().tolist()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_hash": config_hash(cfg) if cfg else None,
        "software": _software_info(),
        "data_sha256": data_hashes,
        "note": "所有数值来自本次实际运行；动态事件为合成场景，不是官方动态基准。",
    }
    with open(os.path.join(summary_dir, "experiment_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {"runs": df, "descriptive": desc, "overall": overall}

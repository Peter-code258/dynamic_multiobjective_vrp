"""命令行入口。

用法示例：
  python -m src.cli prepare --config config/quick.yaml
  python -m src.cli run --config config/quick.yaml
  python -m src.cli analyze --output results/quick
  python -m src.cli all --config config/quick.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from .experiment_runner import (
    analyze_results,
    load_config,
    prepare_data,
    resolve,
    run_experiment,
)


def _config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/quick.yaml", help="YAML 配置文件路径")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="动态多目标车辆路径规划实验平台")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="生成小规模实例与动态事件日程")
    _config_arg(p_prepare)

    p_run = sub.add_parser("run", help="运行 S0-S4 场景实验并导出原始结果")
    _config_arg(p_run)
    p_run.add_argument("--instances", nargs="*", default=None)
    p_run.add_argument("--scenarios", nargs="*", default=None)
    p_run.add_argument("--algorithms", nargs="*", default=None)
    p_run.add_argument("--seeds", nargs="*", type=int, default=None)
    p_run.add_argument("--max-evals", type=int, default=None)
    p_run.add_argument("--output", default=None)
    p_run.add_argument("--no-prepare", action="store_true")
    p_run.add_argument("--no-analyze", action="store_true")

    p_analyze = sub.add_parser("analyze", help="从原始 JSON 生成汇总、统计检验与图表")
    p_analyze.add_argument("--output", required=True)
    p_analyze.add_argument("--config", default=None)
    p_analyze.add_argument("--no-figures", action="store_true")

    p_all = sub.add_parser("all", help="prepare + run + analyze")
    _config_arg(p_all)
    p_all.add_argument("--max-evals", type=int, default=None)
    p_all.add_argument("--output", default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare":
        cfg = load_config(args.config)
        instance_paths, scenario_paths = prepare_data(cfg)
        print("实例: %d 个" % len(instance_paths))
        for path in instance_paths.values():
            print("  " + path)
        print("场景: %d 个" % len(scenario_paths))
        for path in scenario_paths[:5]:
            print("  " + path)
        return 0

    if args.command == "run":
        cfg = load_config(args.config)
        output = args.output or str(cfg.get("experiment", {}).get("output_dir", "results/default"))
        df = run_experiment(
            cfg=cfg,
            instances=args.instances,
            scenarios=args.scenarios,
            algorithms=args.algorithms,
            seeds=args.seeds,
            max_evaluations=args.max_evals,
            output_dir=output,
            prepare=not args.no_prepare,
        )
        print("完成运行: %d 条" % len(df))
        if not args.no_analyze and not df.empty:
            analyze_results(output, cfg=cfg)
            print("汇总与图表: %s" % resolve(output))
        return 0

    if args.command == "analyze":
        cfg = load_config(args.config) if args.config else None
        result = analyze_results(args.output, cfg=cfg, make_figures=not args.no_figures)
        print("汇总完成: %d 条运行" % len(result["runs"]))
        return 0

    if args.command == "all":
        cfg = load_config(args.config)
        output = args.output or str(cfg.get("experiment", {}).get("output_dir", "results/default"))
        prepare_data(cfg)
        df = run_experiment(
            cfg=cfg,
            max_evaluations=args.max_evals,
            output_dir=output,
            prepare=False,
        )
        if not df.empty:
            analyze_results(output, cfg=cfg)
        print("全部完成: %s" % resolve(output))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

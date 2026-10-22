# 动态多目标车辆路径规划（Dynamic Multi-Objective VRP）

面向新增订单、车辆故障与路网拥堵的事件驱动滚动重规划原型。项目在同一套数据模型、约束检查器、四目标 evaluator、仿真器与评价预算下，比较：

- **NSGA-II**：快速非支配排序、拥挤距离、父子合并精英保留。
- **离散 MOPSO**：路线集合编码 + 离散变换速度（插入、交换、2-opt、跨路线迁移、拆分/合并），外部 Pareto 档案。
- **Pareto 档案型 MOACO**：四组目标信息素（每个目标只由该目标上的档案精英解更新）、Dirichlet 随机目标权重、非支配外部档案、按目标覆盖与稀疏度更新精英信息素；蚂蚁随机选择启用车队规模并允许提前关闭路线，以保留“少用车 vs 低响应”折中。
- **动态贪心插入基线**：只用于参照，不参与 HV/IGD 参考前沿构造与算法间统计检验。

> 动态事件是**本项目合成的可复现测试场景**，不是官方动态 VRP 基准。项目不预设任何算法优劣，所有数值均由实际运行产生。

---

## 1. 安装

```powershell
cd dynamic_multiobjective_vrp
python -m pip install -r requirements.txt
```

开发实测环境：Python 3.9.13（Windows）。代码使用 `from __future__ import annotations`，未使用 3.10+ 专有语法。

## 2. 一键运行

```powershell
# 快速流程：生成实例与事件、运行 4 种算法、导出汇总与图表
python -m src.cli all --config config/quick.yaml --output results/quick

# 分步执行
python -m src.cli prepare --config config/quick.yaml
python -m src.cli run --config config/quick.yaml --output results/quick
python -m src.cli analyze --output results/quick
```

常用参数：

```powershell
python -m src.cli run --config config/quick.yaml `
  --instances toy_5 --scenarios S0 S2 `
  --algorithms nsga2 mopso moaco greedy `
  --seeds 11 12 --max-evals 800 --output results/probe
```

正式配置见 `config/default.yaml`、`config/full.yaml`；消融、敏感性与规模实验示例见 `config/ablation.yaml`、`config/sensitivity.yaml`、`config/scale.yaml`。计算量较大，建议先跑 `quick.yaml`。

## 3. 目录结构

```text
dynamic_multiobjective_vrp/
├── README.md
├── requirements.txt
├── config/                     # quick / default / full / 消融 / 敏感性 / 规模
├── data/
│   ├── instances/              # 合成 JSON 实例；可放 Solomon 文本实例
│   └── scenarios/              # 每个（实例, 场景, 种子）一份预生成事件 JSON
├── src/
│   ├── models.py               # Customer / Vehicle / VRPInstance / DynamicEvent / Scenario
│   ├── instance_loader.py      # JSON 与 Solomon 读取、场景读写
│   ├── travel_time.py          # 距离缓存、时变行驶时间、重叠拥堵取最大倍率
│   ├── context.py              # 统一规划上下文与剩余容量
│   ├── constraints.py          # 统一结构/容量/可用性/未服务检查
│   ├── objectives.py           # 唯一四目标 evaluator 与诊断指标
│   ├── solution.py             # 路线编码、解码、修复、交叉、变异、贪心构造
│   ├── pareto.py               # 支配、约束支配、快速非支配排序、档案
│   ├── algorithms/             # nsga2 / mopso / moaco + 注册入口
│   ├── baselines/              # 贪心插入基线
│   ├── simulator.py            # 事件驱动滚动重规划、冻结、恢复指标
│   ├── metrics.py              # HV、IGD/IGD+、共享归一化
│   ├── statistics.py           # Wilcoxon、Friedman、Holm 校正
│   ├── visualization.py        # 路线图、Pareto 投影、统计图
│   ├── experiment_runner.py    # 数据准备、统一运行、汇总与图表
│   └── cli.py                  # prepare / run / analyze / all
├── experiments/run_experiments.py
├── tests/                      # 64 项单元测试
└── results/                    # raw / summary / figures
```

## 4. 数据格式

实例 JSON：

```json
{
  "instance_id": "toy_5",
  "depot": {"id": 0, "x": 50.0, "y": 50.0},
  "vehicles": [{"vehicle_id": 0, "capacity": 40.0}],
  "customers": [
    {
      "customer_id": 1, "x": 20.0, "y": 30.0, "demand": 8.0,
      "service_time": 3.0, "release_time": 0.0,
      "ready_time": 0.0, "due_time": null
    }
  ],
  "distance_type": "euclidean",
  "metadata": {}
}
```

场景 JSON 由生成器写出，包含 `scenario_id`、`seed`、`events`、`intensity`、`metadata`。事件类型：

- `NEW_ORDER`：payload 内含新客户坐标、需求、服务时长与 `release_time`。
- `VEHICLE_BREAKDOWN`：`vehicle_id`，故障后该车立即不可用，未完成客户回到待分配集合。
- `CONGESTION_START` / `CONGESTION_END`：受影响无向边、行驶时间倍率、持续时间；只改时间，不改距离。

`prepare` 会为每个 **（实例 × 场景 × 运行种子）** 生成 `data/scenarios/{实例}__{场景}__s{种子}.json`。同一组合下三种算法读取完全相同的事件文件；事件随机流由 `scenario_seed(base_seed, run_seed, spec_index)` 独立决定，与算法随机流分离。

## 5. 四个目标与口径

全部最小化，统一由 `src/objectives.py` 计算：

1. `f1_total_distance`：实际执行弧段距离累计 + 收工返回配送中心的距离；重规划不会抹掉已完成距离。故障车辆已执行距离保留，但不计返仓距离。
2. `f2_used_vehicles`：规划周期内启用过的不同车辆数（故障前已执行任务的车仍计入）。
3. `f3_total_response_time`：订单从 `release_time` 到开始服务的时长之和。
4. `f4_load_utilization_variance`：**已启用车辆**容量利用率方差。

口径说明（开题报告 md 与实验设计方案 docx 存在差异）：

- md 规定 f3 为“订单响应时长”，本项目主目标按 md 实现；另报 `total_lateness`（晚于 due_time）、`total_early_wait`（早于 ready_time）与 `total_service_wait`（到达至开始服务），三者不可混为一谈。
- md 规定 f4 只统计已启用车辆；docx 建议统计所有可用车辆。本项目主目标按 md，另报 `load_utilization_variance_all_available` 与 `load_variance_all_available` 作为辅助口径。
- Solomon 的 `ready_time/due_time` 视为时间窗，`release_time` 固定为 0；静态订单不会被错误地当作未来才释放。

## 6. 动态滚动重规划

动态响应策略可通过 `planner.response_strategies` 切换，并记录在每次运行与重规划日志中：

- `restart`：每次重规划从零开始，不使用历史方案。
- `local_repair`：只把上一条执行方案修复后作为初始种子。
- `warm_start`：上一条方案 + 上轮 Pareto 档案前 3 个解作为种子。
- `mixed`：历史方案 + 更多档案解 + 变异副本作为种子。

- 已完成客户、已执行距离与响应时长锁定；正在行驶的弧段按“出发即完整执行该步”处理，事件规划时刻可能略晚于名义事件时刻，记录为 `planning_time`。
- 故障车辆立即禁止接单；其未完成客户返回 `PENDING`。
- 所有候选方案共享同一约束检查器与 evaluator；容量按“已服务累计需求 + 本段需求”检查。
- 评价预算 `max_evaluations` 是**每次重规划调用**的预算，四种算法相同；`evaluations_total` 为整个动态仿真累计值。
- 若运力不足，保留未服务订单并写入 `unserved_orders`、`service_rate`、`feasible=false`，不伪造全服务解。
- 执行策略固定为归一化理想点折中解；完整 Pareto 档案仍保留用于质量评价。
- 每条重规划日志记录：受影响客户/车辆/边、路网版本、可用车辆数、车辆载荷、未服务数、评价次数、耗时与路线修改率。
- 恢复指标：事件发生到受影响订单首次全部 `ASSIGNED/SERVED` 的模拟时延与评价次数；无法恢复的事件显式记为 `recovered=false`。无事件的场景 `recovery_events=0`，`recovery_rate` 取空集约定值 1.0，不能据此比较算法。

## 7. 输出

```text
results/<name>/
├── raw/*.json                     # 每次运行：配置、指标、重规划日志、恢复记录、Pareto 档案、最终路线
├── summary/
│   ├── runs.csv                   # 运行级明细
│   ├── vehicle_tasks.csv          # 每次运行每辆车的任务、载荷与距离
│   ├── descriptive_by_scenario.csv
│   ├── descriptive_overall.csv
│   ├── wilcoxon_pairwise.csv      # 配对 Wilcoxon + Holm 校正
│   ├── friedman_tests.csv
│   └── experiment_manifest.json
├── figures/
│   ├── box_*.png
│   ├── line_*.png
│   ├── pareto_*.png
│   ├── route_*.png
│   └── route_breakdown_*.png      # 故障前后路线对比
└── failures.json                  # 仅在存在失败运行时生成
```

HV/IGD 说明：同一（实例, 场景）内使用共享归一化上下界，并以 `nsga2/mopso/moaco` 所有种子最终档案的并集非支配解作为经验参考前沿；贪心基线不参与参考前沿与算法间统计检验。参考前沿是经验近似，不是已知真实 Pareto 前沿。

## 8. 算法说明与已知局限

- **MOACO 不是唯一标准算法。** 本项目实现的是“Pareto 档案 + 四组目标信息素 + 随机目标权重”的明确变体，规则公开在代码与参数中，不代表所有 MOACO。
- 动态事件为合成测试场景；订单坐标、到达时刻、故障车辆与拥堵路段均由固定种子生成。
- 时变行驶时间为分段常数 FIFO 近似；同一时刻重叠拥堵取最大倍率，不累乘。
- 第一版为抽象任务分配模型，不模拟仓库分拣、装车与补货；不支持返仓多趟。
- 只支持一个配送中心；主实验使用同质车辆，但数据层允许异构容量。
- 运行耗时指标（`replan_time_mean/max`）取决于机器与负载，不属于可复现性保证；目标值、路线与事件日程在同种子下可复现。
- `toy_5` 只有 5 个客户，Pareto 前沿常塌缩为单点，三种算法会收敛到同一折中解，因此它只用于流程验证；算法差异请在 `toy_10` 及以上实例观察。
- 实验脚本不会生成虚构结果；未运行的配置不会出现在输出中。

## 9. 测试

```powershell
python -m pytest -q
```

覆盖：四目标计算与容量累计口径、约束检查、Pareto 排序与档案、HV/IGD、统计检验、三种算法与贪心基线、动态响应策略、动态事件（新增订单/故障/拥堵/组合）、容量不突破、同种子可复现与事件日程共享。

## 10. 参考文献（实现依据）

- Deb et al., 2002, NSGA-II, IEEE TEC. https://doi.org/10.1109/4235.996017
- Coello Coello et al., 2004, MOPSO, IEEE TEC. https://doi.org/10.1109/TEVC.2004.826067
- Pillac et al., 2013, Dynamic VRP review, EJOR. https://doi.org/10.1016/j.ejor.2012.08.015
- Ojeda Rios et al., 2021, Recent dynamic VRP review, CIE. https://doi.org/10.1016/j.cie.2021.107604
- Blauth et al., 2024, Time-dependent VRP benchmarks, Discrete Optimization. https://doi.org/10.1016/j.disopt.2024.100848
- Figliozzi, 2012, Time-dependent congestion VRP, TRE. https://doi.org/10.1016/j.tre.2011.11.006

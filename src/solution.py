"""路线编码、修复、交叉与离散变异算子。

统一表示：{vehicle_id: [customer_id, ...]}，不显式保存仓库节点。
所有算子只保证结构与容量，最终可行性由统一评估器复核。
"""
from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .context import PlanningContext, remaining_capacity


Routes = Dict[int, List[int]]


def clone_routes(routes: Routes) -> Routes:
    return {int(vid): list(route) for vid, route in routes.items() if route}


def route_load(route: Iterable[int], ctx: PlanningContext) -> float:
    total = 0.0
    for cid in route:
        customer = ctx.instance.customers.get(cid)
        if customer is not None:
            total += customer.demand
    return total


def route_distance(route: Sequence[int], start_node: int, ctx: PlanningContext) -> float:
    node = start_node
    d = 0.0
    for cid in route:
        d += ctx.travel.distance(node, cid)
        node = cid
    d += ctx.travel.distance(node, ctx.instance.depot_id)
    return d


def _valid_ids(ctx: PlanningContext) -> set:
    return set(ctx.remaining)


def _clean(routes: Routes, ctx: PlanningContext) -> Tuple[Routes, List[int]]:
    """删除不可用车辆、重复/已服务客户，并返还未被覆盖的客户。"""
    valid = _valid_ids(ctx)
    available = set(ctx.available_vehicle_ids)
    out: Routes = {}
    seen: set = set()
    for vid, route in routes.items():
        if vid not in available:
            continue
        cleaned: List[int] = []
        for cid in route:
            if cid in valid and cid not in seen:
                cleaned.append(cid)
                seen.add(cid)
        if cleaned:
            out[vid] = cleaned
    missing = [cid for cid in ctx.remaining if cid not in seen]
    return out, missing


def _best_insertion(
    routes: Routes,
    cid: int,
    ctx: PlanningContext,
    rng: Optional[np.random.Generator] = None,
):
    customer = ctx.instance.customers[cid]
    best = None  # (cost, vid, pos)
    for vid, route in routes.items():
        view = ctx.vehicles[vid]
        remaining_cap = remaining_capacity(view)
        if route_load(route, ctx) + customer.demand > remaining_cap + 1e-9:
            continue
        base = route_distance(route, view.start_node, ctx)
        node = view.start_node
        # 逐位置计算增量距离
        prefix = 0.0
        for pos in range(len(route) + 1):
            if pos == 0:
                before = node
                after = route[0]
            else:
                before = route[pos - 1]
                after = route[pos] if pos < len(route) else ctx.instance.depot_id
            delta = (
                ctx.travel.distance(before, cid)
                + ctx.travel.distance(cid, after)
                - ctx.travel.distance(before, after)
            )
            cost = base + delta
            if rng is not None:
                cost *= 1.0 + 0.05 * float(rng.random())
            if best is None or cost < best[0]:
                best = (cost, vid, pos)
    return best


def _new_route_cost(vid: int, cid: int, ctx: PlanningContext, rng: Optional[np.random.Generator]):
    view = ctx.vehicles[vid]
    cost = (
        ctx.travel.distance(view.start_node, cid)
        + ctx.travel.distance(cid, ctx.instance.depot_id)
        - ctx.travel.distance(view.start_node, ctx.instance.depot_id)
    )
    if rng is not None:
        cost *= 1.0 + 0.05 * float(rng.random())
    return cost


def repair_routes(
    routes: Routes,
    ctx: PlanningContext,
    rng: Optional[np.random.Generator] = None,
) -> Routes:
    """容量修复 + 缺失客户贪心插入。

    无法在可用容量内安排的客户会被保留为未服务，由评估器记为不可行；
    这里不会静默删除客户，也不会伪造全服务方案。
    """
    cleaned, missing = _clean(routes, ctx)
    available = list(ctx.available_vehicle_ids)

    # 先将超载路线尾部客户挤出，交给缺失集合统一重插。
    rebuilt: Routes = {}
    overflow: List[int] = []
    for vid, route in cleaned.items():
        cap = remaining_capacity(ctx.vehicles[vid])
        load = 0.0
        seq: List[int] = []
        for cid in route:
            demand = ctx.instance.customers[cid].demand
            if load + demand <= cap + 1e-9:
                seq.append(cid)
                load += demand
            else:
                overflow.append(cid)
        if seq:
            rebuilt[vid] = seq
    missing.extend(overflow)
    # 去重（一个客户可能既被挤出又在 missing 中）
    missing = sorted(set(missing), key=lambda c: (
        ctx.instance.customers[c].release_time,
        -ctx.instance.customers[c].demand,
        c,
    ))

    unused = [vid for vid in available if vid not in rebuilt]
    for cid in missing:
        insert = _best_insertion(rebuilt, cid, ctx, rng) if rebuilt else None
        if insert is not None:
            _, vid, pos = insert
            rebuilt[vid].insert(pos, cid)
            continue
        if unused:
            fit = [
                v for v in unused
                if ctx.instance.customers[cid].demand
                <= remaining_capacity(ctx.vehicles[v]) + 1e-9
            ]
            if fit:
                best_vid = min(fit, key=lambda v: _new_route_cost(v, cid, ctx, rng))
                rebuilt[best_vid] = [cid]
                unused.remove(best_vid)
                continue
        # 容量与车辆都用尽：保留为未服务（不丢弃、不伪造）。
    return clone_routes(rebuilt)


def decode_permutation(
    order: Sequence[int],
    ctx: PlanningContext,
    rng: Optional[np.random.Generator] = None,
) -> Routes:
    """共享解码器：客户排列 -> 车辆路线集合。"""
    available = list(ctx.available_vehicle_ids)
    routes: Routes = {}
    idx = 0
    for vid in available:
        if idx >= len(order):
            break
        cap = remaining_capacity(ctx.vehicles[vid])
        load = 0.0
        seq: List[int] = []
        while idx < len(order):
            cid = order[idx]
            demand = ctx.instance.customers[cid].demand
            if load + demand > cap + 1e-9 and seq:
                break
            if load + demand > cap + 1e-9 and not seq:
                idx += 1
                continue
            seq.append(cid)
            load += demand
            idx += 1
        if seq:
            routes[vid] = seq
    if idx < len(order):
        # 车辆数不足：剩余客户交给修复器，最终由评估器记录未服务。
        pass
    return repair_routes(routes, ctx, rng)


def random_solution(ctx: PlanningContext, rng: np.random.Generator) -> Routes:
    order = list(ctx.remaining)
    rng.shuffle(order)
    return decode_permutation(order, ctx, rng)


def greedy_solution(ctx: PlanningContext, rng: Optional[np.random.Generator] = None) -> Routes:
    """最近邻构造：从仓库/当前位置逐步扩展，容量不足则启用下一辆车。"""
    available = list(ctx.available_vehicle_ids)
    unassigned = set(ctx.remaining)
    routes: Routes = {}
    for vid in available:
        if not unassigned:
            break
        view = ctx.vehicles[vid]
        node = view.start_node
        cap = remaining_capacity(view)
        load = 0.0
        seq: List[int] = []
        while unassigned:
            feasible = [
                c for c in unassigned
                if load + ctx.instance.customers[c].demand <= cap + 1e-9
            ]
            if not feasible:
                break
            feasible.sort(key=lambda c: (ctx.travel.distance(node, c), c))
            k = 0
            if rng is not None and len(feasible) > 1 and rng.random() < 0.20:
                k = int(rng.integers(0, min(3, len(feasible))))
            cid = feasible[k]
            seq.append(cid)
            load += ctx.instance.customers[cid].demand
            unassigned.remove(cid)
            node = cid
        if seq:
            routes[vid] = seq
    return repair_routes(routes, ctx, rng)


def crossover_routes(
    parent_a: Routes,
    parent_b: Routes,
    ctx: PlanningContext,
    rng: np.random.Generator,
) -> Routes:
    """以路线为单位的顺序交叉：保留 A 的部分整条路线，再从 B 补充剩余客户。"""
    child: Routes = {}
    seen: set = set()
    a_items = list(parent_a.items())
    b_items = list(parent_b.items())
    rng.shuffle(a_items)
    keep = max(1, len(a_items) // 2)
    for vid, route in a_items[:keep]:
        copied = [c for c in route if c not in seen]
        if copied:
            child[vid] = copied
            seen.update(copied)
    available = set(ctx.available_vehicle_ids)
    # 从 B 中按路线顺序补充未被继承的客户
    for vid_b, route_b in b_items:
        for cid in route_b:
            if cid in seen:
                continue
            target = vid_b if vid_b in available and vid_b not in child else None
            if target is None and child:
                target = list(child.keys())[int(rng.integers(0, len(child)))]
            if target is None:
                target = next((v for v in ctx.available_vehicle_ids if v not in child), None)
            if target is None:
                break
            child.setdefault(target, []).append(cid)
            seen.add(cid)
    return repair_routes(child, ctx, rng)


def _routes_list(routes: Routes) -> List[Tuple[int, List[int]]]:
    return [(vid, list(route)) for vid, route in routes.items() if route]


def mutate_routes(
    routes: Routes,
    ctx: PlanningContext,
    rng: np.random.Generator,
    strength: int = 1,
) -> Routes:
    """离散变异：插入、交换、2-opt、跨路线迁移、拆分/合并、路线顺序扰动。"""
    out = clone_routes(routes)
    for _ in range(max(1, strength)):
        op = int(rng.integers(0, 6))
        items = _routes_list(out)
        if not items:
            return repair_routes(out, ctx, rng)
        if op == 0 and len(ctx.remaining) >= 2:
            # 交换两个客户（可跨路线）
            all_customers = [(vid, i) for vid, r in out.items() for i in range(len(r))]
            if len(all_customers) >= 2:
                a, b = rng.choice(len(all_customers), size=2, replace=False)
                va, ia = all_customers[int(a)]
                vb, ib = all_customers[int(b)]
                out[va][ia], out[vb][ib] = out[vb][ib], out[va][ia]
        elif op == 1:
            # 把客户迁移到另一位置/路线
            vids = [v for v, r in out.items() if r]
            if len(vids) >= 1:
                src = vids[int(rng.integers(0, len(vids)))]
                pos = int(rng.integers(0, len(out[src])))
                cid = out[src].pop(pos)
                targets = [v for v in ctx.available_vehicle_ids if v != src]
                if targets and rng.random() < 0.6:
                    dst = targets[int(rng.integers(0, len(targets)))]
                    out.setdefault(dst, [])
                    out[dst].insert(int(rng.integers(0, len(out[dst]) + 1)), cid)
                else:
                    out[src].insert(int(rng.integers(0, len(out[src]) + 1)), cid)
                if not out[src]:
                    out.pop(src, None)
        elif op == 2:
            # 2-opt：同一路线内反转片段
            vids = [v for v, r in out.items() if len(r) >= 2]
            if vids:
                vid = vids[int(rng.integers(0, len(vids)))]
                r = out[vid]
                i, j = sorted(rng.choice(len(r), size=2, replace=False))
                r[i : j + 1] = list(reversed(r[i : j + 1]))
        elif op == 3:
            # 拆分一条较长路线
            long_vids = [v for v, r in out.items() if len(r) >= 3]
            unused = [v for v in ctx.available_vehicle_ids if v not in out]
            if long_vids and unused:
                vid = long_vids[int(rng.integers(0, len(long_vids)))]
                cut = 1 + int(rng.integers(0, max(1, len(out[vid]) - 1)))
                new_vid = unused[0]
                out[new_vid] = out[vid][cut:]
                out[vid] = out[vid][:cut]
        elif op == 4:
            # 合并两条路线
            vids = [v for v, r in out.items() if r]
            if len(vids) >= 2:
                v1, v2 = rng.choice(vids, size=2, replace=False)
                merged = out[v1] + out[v2]
                if route_load(merged, ctx) <= remaining_capacity(ctx.vehicles[v1]) + 1e-9:
                    out[v1] = merged
                    out.pop(v2, None)
        else:
            # 路线顺序重排（等价于客户顺序扰动，不会丢客户）
            for vid in list(out.keys()):
                if len(out[vid]) >= 2 and rng.random() < 0.5:
                    i, j = sorted(rng.choice(len(out[vid]), size=2, replace=False))
                    out[vid][i], out[vid][j] = out[vid][j], out[vid][i]
    return repair_routes(out, ctx, rng)

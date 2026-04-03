"""
Entry Filter Backtest — 进场条件网格搜索

对 trades.csv 中的历史交易进行事后过滤，模拟"如果当初设置更严格的进场门槛，
结果会变好还是变坏"，从而找到单位交易收益最高的参数组合。

可搜索的进场维度：
    entry_score   — 慢层 unified alpha score（即慢层 slow_score）
                    LONG 为正，SHORT 为负；取绝对值代表信号强度
    open_vol      — 进场时的近期波动率（Layer 1 动态止损基准）
    side          — 方向过滤（ALL / LONG / SHORT）

注：fast_score（LOBTimingEngine timing_score）目前未写入 trades.csv。
    如需加入 fast_score 过滤，请在 alpha_strategy._open_long/_open_short 中
    将 timing_score 存入 pos dict，并在 trade_recorder.open() 调用时传入。

输出：
    控制台打印 Top N 参数组合（按 avg_net_pnl 排序）
    results_entry_filter.csv — 全部参数组合的完整结果

用法：
    cd institutional_crypto_quant
    python live_trading/entry_filter_backtest.py
    python live_trading/entry_filter_backtest.py --csv trades.csv --top 30 --min-trades 10
"""

import argparse
import csv
import itertools
import math
import os
import sys
from typing import List, Dict, Any


# ── 网格参数定义 ──────────────────────────────────────────────────────────────

GRID = {
    # 信号强度下限：|entry_score| 必须高于此值才进场（对应 slow_score 门槛）
    "min_abs_score": [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0],

    # 波动率上限：open_vol 必须低于此值（过滤极端波动环境）
    "max_open_vol":  [0.005, 0.008, 0.010, 0.015, 0.020, 0.025],

    # 波动率下限：open_vol 必须高于此值（过滤流动性太差、几乎不动的品种）
    "min_open_vol":  [0.0, 0.001, 0.002],

    # 方向过滤
    "side":          ["ALL", "LONG", "SHORT"],
}

# 结果输出文件（相对于 trades.csv 所在目录）
RESULT_CSV = "results_entry_filter.csv"

# 结果表列定义
RESULT_COLS = [
    "min_abs_score", "max_open_vol", "min_open_vol", "side",
    "n_trades",
    "total_net_pnl", "avg_net_pnl",
    "win_rate",
    "profit_factor",
    "avg_hold_seconds",
    "best_exit_reason",          # 该子集中最常见的 exit reason
    "sharpe_proxy",              # avg_pnl / std_pnl（正态假设下的简化 Sharpe）
]


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_trades(csv_path: str) -> List[Dict[str, Any]]:
    """加载 trades.csv，返回类型转换后的记录列表"""
    if not os.path.exists(csv_path):
        print(f"[ERROR] 文件不存在: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "trade_id":             r["trade_id"],
                    "symbol":               r["symbol"],
                    "side":                 r["side"],
                    "entry_score":          float(r["entry_score"]),
                    "abs_score":            abs(float(r["entry_score"])),
                    "open_vol":             float(r["open_vol"]),
                    "exit_reason":          r["exit_reason"],
                    "hold_seconds":         float(r["hold_seconds"]),
                    "net_pnl":              float(r["net_pnl"]),
                    "ret_pct":              float(r["ret_pct"]),
                    "ret_lev_pct":          float(r["ret_lev_pct"]),
                    "lc_state_at_exit":     r.get("lc_state_at_exit", ""),
                    "aligned_score_at_exit": float(r.get("aligned_score_at_exit", 0)),
                    "velocity_at_exit":     float(r.get("velocity_at_exit", 0)),
                })
            except (ValueError, KeyError):
                continue  # 跳过损坏行
    return rows


# ── 统计计算 ──────────────────────────────────────────────────────────────────

def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _most_common(items: List[str]) -> str:
    if not items:
        return ""
    freq: Dict[str, int] = {}
    for x in items:
        freq[x] = freq.get(x, 0) + 1
    return max(freq, key=freq.get)


def compute_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """计算一组交易的核心统计指标"""
    n = len(trades)
    if n == 0:
        return None

    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]

    total_pnl  = sum(pnls)
    avg_pnl    = total_pnl / n
    win_rate   = len(wins) / n

    profit_sum = sum(wins)
    loss_sum   = abs(sum(loss)) if loss else 0.0
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else float("inf")

    avg_hold   = sum(t["hold_seconds"] for t in trades) / n
    std_pnl    = _std(pnls)
    sharpe     = avg_pnl / std_pnl if std_pnl > 0 else 0.0

    best_exit  = _most_common([t["exit_reason"] for t in trades])

    return {
        "n_trades":         n,
        "total_net_pnl":    round(total_pnl, 4),
        "avg_net_pnl":      round(avg_pnl, 5),
        "win_rate":         round(win_rate, 4),
        "profit_factor":    round(profit_factor, 3),
        "avg_hold_seconds": round(avg_hold, 1),
        "best_exit_reason": best_exit,
        "sharpe_proxy":     round(sharpe, 4),
    }


# ── 网格搜索 ──────────────────────────────────────────────────────────────────

def run_grid_search(
    trades: List[Dict],
    min_trades: int = 5,
) -> List[Dict[str, Any]]:
    """
    遍历所有参数组合，过滤子集并计算统计指标。

    min_trades: 子集交易数量低于此值则跳过（样本太少，结果不可信）
    """
    keys   = list(GRID.keys())
    values = [GRID[k] for k in keys]

    results = []
    total_combos = 1
    for v in values:
        total_combos *= len(v)

    print(f"[网格搜索] 参数组合总数: {total_combos}  全量交易: {len(trades)} 笔")
    print(f"[网格搜索] 最小子集交易数: {min_trades}")
    print()

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        # ── 过滤条件 ──────────────────────────────────────────────────────────
        subset = [
            t for t in trades
            if (
                t["abs_score"]   >= params["min_abs_score"]
                and t["open_vol"] <= params["max_open_vol"]
                and t["open_vol"] >= params["min_open_vol"]
                and (params["side"] == "ALL" or t["side"] == params["side"])
            )
        ]

        if len(subset) < min_trades:
            continue

        metrics = compute_metrics(subset)
        if metrics is None:
            continue

        results.append({**params, **metrics})

    return results


# ── 输出 ──────────────────────────────────────────────────────────────────────

def print_top_results(results: List[Dict], top_n: int):
    """按 avg_net_pnl 降序打印 Top N 结果"""
    sorted_r = sorted(results, key=lambda x: x["avg_net_pnl"], reverse=True)
    top = sorted_r[:top_n]

    # 列宽对齐
    COL_W = {
        "min_abs_score": 13,
        "max_open_vol":  12,
        "min_open_vol":  12,
        "side":           6,
        "n_trades":       8,
        "total_net_pnl": 12,
        "avg_net_pnl":   12,
        "win_rate":       9,
        "profit_factor": 13,
        "sharpe_proxy":  12,
        "best_exit_reason": 22,
    }

    header_cols = [
        "min_abs_score", "max_open_vol", "min_open_vol", "side",
        "n_trades", "total_net_pnl", "avg_net_pnl", "win_rate",
        "profit_factor", "sharpe_proxy", "best_exit_reason",
    ]

    header = "  ".join(c.ljust(COL_W[c]) for c in header_cols)
    sep    = "-" * len(header)

    print(f"\n{'='*80}")
    print(f"  Top {top_n} 参数组合（按 avg_net_pnl 排序 — 单位交易收益最高）")
    print(f"{'='*80}")
    print(header)
    print(sep)

    for r in top:
        row = "  ".join(
            str(r.get(c, "")).ljust(COL_W[c]) for c in header_cols
        )
        print(row)

    print()

    # 特别标注最优组合
    best = sorted_r[0]
    print("★  最优参数组合（单位交易收益最高）：")
    for k in ["min_abs_score", "max_open_vol", "min_open_vol", "side"]:
        print(f"    {k:20s} = {best[k]}")
    print(f"    {'交易数':20s} = {best['n_trades']}")
    print(f"    {'总净PnL (USDT)':20s} = {best['total_net_pnl']}")
    print(f"    {'平均净PnL (USDT)':20s} = {best['avg_net_pnl']}")
    print(f"    {'胜率':20s} = {best['win_rate']:.1%}")
    print(f"    {'盈亏比':20s} = {best['profit_factor']}")
    print(f"    {'Sharpe (简化)':20s} = {best['sharpe_proxy']}")
    print(f"    {'主要退出原因':20s} = {best['best_exit_reason']}")

    # 附加：按 Sharpe 最优
    best_sharpe = max(results, key=lambda x: x["sharpe_proxy"])
    if best_sharpe["min_abs_score"] != best["min_abs_score"] or best_sharpe["side"] != best["side"]:
        print()
        print("★  最优参数组合（Sharpe 最高）：")
        for k in ["min_abs_score", "max_open_vol", "min_open_vol", "side"]:
            print(f"    {k:20s} = {best_sharpe[k]}")
        print(f"    {'平均净PnL (USDT)':20s} = {best_sharpe['avg_net_pnl']}")
        print(f"    {'Sharpe (简化)':20s} = {best_sharpe['sharpe_proxy']}")
        print(f"    {'胜率':20s} = {best_sharpe['win_rate']:.1%}")

    # 附加：按 win_rate 最优
    best_wr = max(results, key=lambda x: x["win_rate"])
    print()
    print("★  最优参数组合（胜率最高）：")
    for k in ["min_abs_score", "max_open_vol", "min_open_vol", "side"]:
        print(f"    {k:20s} = {best_wr[k]}")
    print(f"    {'交易数':20s} = {best_wr['n_trades']}")
    print(f"    {'胜率':20s} = {best_wr['win_rate']:.1%}")
    print(f"    {'平均净PnL (USDT)':20s} = {best_wr['avg_net_pnl']}")

    print()


def print_sensitivity(results: List[Dict]):
    """
    边际分析：固定其他参数为最优值时，各参数单独变化对 avg_pnl 的影响。
    帮助判断哪个参数对结果最敏感。
    """
    print(f"{'='*60}")
    print("  边际敏感性分析（各维度均值 avg_net_pnl）")
    print(f"{'='*60}")

    for key in ["min_abs_score", "max_open_vol", "min_open_vol", "side"]:
        groups: Dict[Any, List[float]] = {}
        for r in results:
            v = r[key]
            groups.setdefault(v, []).append(r["avg_net_pnl"])

        print(f"\n  {key}:")
        for val in sorted(groups.keys(), key=lambda x: str(x)):
            vals = groups[val]
            mean_pnl = sum(vals) / len(vals)
            n_combos = len(vals)
            bar = "█" * max(1, int((mean_pnl + 0.002) * 2000))
            print(f"    {str(val):8s}  avg_pnl={mean_pnl:+.5f}  (n={n_combos})  {bar}")
    print()


def save_results(results: List[Dict], out_path: str):
    """将全部参数组合结果保存为 CSV"""
    if not results:
        return
    sorted_r = sorted(results, key=lambda x: x["avg_net_pnl"], reverse=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLS)
        writer.writeheader()
        for r in sorted_r:
            writer.writerow({c: r.get(c, "") for c in RESULT_COLS})
    print(f"[输出] 完整结果已保存到: {out_path}  ({len(sorted_r)} 组)")


# ── 快速诊断：不过滤时的基准 ─────────────────────────────────────────────────

def print_baseline(trades: List[Dict]):
    """打印全量交易的基准统计（无过滤时的起点）"""
    m = compute_metrics(trades)
    print(f"{'='*60}")
    print("  全量交易基准统计（无任何过滤）")
    print(f"{'='*60}")
    print(f"  交易笔数:          {m['n_trades']}")
    print(f"  总净PnL (USDT):    {m['total_net_pnl']}")
    print(f"  平均净PnL (USDT):  {m['avg_net_pnl']}")
    print(f"  胜率:              {m['win_rate']:.1%}")
    print(f"  盈亏比:            {m['profit_factor']}")
    print(f"  平均持仓(秒):      {m['avg_hold_seconds']}")
    print(f"  Sharpe(简化):      {m['sharpe_proxy']}")
    print()

    # 各 exit_reason 分组
    from collections import defaultdict
    by_reason: Dict[str, List] = defaultdict(list)
    for t in trades:
        by_reason[t["exit_reason"]].append(t["net_pnl"])
    print("  按 exit_reason 分组:")
    for reason, pnls in sorted(by_reason.items()):
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        print(
            f"    {reason:25s}  n={len(pnls):3d}  "
            f"avg={sum(pnls)/len(pnls):+.5f}  "
            f"胜率={wr:.0%}  total={sum(pnls):+.4f}"
        )

    # 各 side 分组
    by_side: Dict[str, List] = defaultdict(list)
    for t in trades:
        by_side[t["side"]].append(t["net_pnl"])
    print("\n  按 side 分组:")
    for side, pnls in sorted(by_side.items()):
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        print(
            f"    {side:6s}  n={len(pnls):3d}  "
            f"avg={sum(pnls)/len(pnls):+.5f}  "
            f"胜率={wr:.0%}  total={sum(pnls):+.4f}"
        )

    # entry_score 分位数分析
    abs_scores = sorted(t["abs_score"] for t in trades)
    pcts = [0.25, 0.50, 0.75, 0.90]
    print("\n  |entry_score| 分位数:")
    for pct in pcts:
        idx = int(pct * len(abs_scores))
        print(f"    P{int(pct*100):2d} = {abs_scores[min(idx, len(abs_scores)-1)]:.4f}")

    # open_vol 分布
    vols = sorted(t["open_vol"] for t in trades)
    print("\n  open_vol 分位数:")
    for pct in pcts:
        idx = int(pct * len(vols))
        print(f"    P{int(pct*100):2d} = {vols[min(idx, len(vols)-1)]:.5f} ({vols[min(idx, len(vols)-1)]:.3%})")
    print()


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="进场过滤回测 — 网格搜索最优进场条件"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="trades.csv 路径（默认：脚本目录上级的 trades.csv）",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="显示 Top N 参数组合（默认 20）",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=5,
        help="过滤子集最少交易数，低于此值跳过（默认 5）",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="跳过边际敏感性分析",
    )
    args = parser.parse_args()

    # 定位 trades.csv
    if args.csv:
        csv_path = args.csv
    else:
        # 默认：脚本位于 live_trading/，trades.csv 在上一级
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path   = os.path.join(script_dir, "..", "trades.csv")
        csv_path   = os.path.normpath(csv_path)

    out_dir  = os.path.dirname(csv_path)
    out_path = os.path.join(out_dir, RESULT_CSV)

    print(f"\n[Entry Filter Backtest]  trades.csv = {csv_path}")
    trades = load_trades(csv_path)
    print(f"[加载完成]  有效记录: {len(trades)} 笔\n")

    # 1. 基准统计
    print_baseline(trades)

    # 2. 网格搜索
    results = run_grid_search(trades, min_trades=args.min_trades)

    if not results:
        print("[WARNING] 没有满足条件的参数组合，请降低 --min-trades 阈值。")
        return

    print(f"[完成]  有效参数组合: {len(results)} 个\n")

    # 3. 输出 Top N
    print_top_results(results, args.top)

    # 4. 边际敏感性分析
    if not args.no_sensitivity:
        print_sensitivity(results)

    # 5. 保存完整结果
    save_results(results, out_path)


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def safe_div(a, b):
    return a / b if b not in (0, 0.0) else 0.0


def win_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return (series > 0).mean()


def summarize_group(df: pd.DataFrame, group_col: str, pnl_col: str = "net_pnl") -> pd.DataFrame:
    if group_col not in df.columns:
        raise ValueError(f"Missing column: {group_col}")

    grouped = df.groupby(group_col, dropna=False)

    out = grouped.agg(
        trades=("trade_id", "count"),
        total_pnl=(pnl_col, "sum"),
        avg_pnl=(pnl_col, "mean"),
        median_pnl=(pnl_col, "median"),
        avg_ret_lev_pct=("ret_lev_pct", "mean"),
        median_ret_lev_pct=("ret_lev_pct", "median"),
        avg_hold_seconds=("hold_seconds", "mean"),
        median_hold_seconds=("hold_seconds", "median"),
        avg_fee=("fee_usdt", "mean"),
        total_fee=("fee_usdt", "sum"),
    ).reset_index()

    wr = grouped[pnl_col].apply(win_rate).reset_index(name="win_rate")
    out = out.merge(wr, on=group_col, how="left")

    out["pnl_per_trade"] = out["total_pnl"] / out["trades"]
    out = out.sort_values(["total_pnl", "win_rate"], ascending=[False, False]).reset_index(drop=True)
    return out


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 持仓时间分桶
    hold_bins = [-np.inf, 30, 60, 90, 120, 180, 300, 600, np.inf]
    hold_labels = [
        "<=30s", "30-60s", "60-90s", "90-120s",
        "120-180s", "180-300s", "300-600s", ">600s"
    ]
    df["hold_bucket"] = pd.cut(df["hold_seconds"], bins=hold_bins, labels=hold_labels)

    # 收益率分桶
    ret_bins = [-np.inf, -10, -5, -2, 0, 2, 5, 10, np.inf]
    ret_labels = ["<-10%", "-10~-5%", "-5~-2%", "-2~0%", "0~2%", "2~5%", "5~10%", ">10%"]
    df["ret_bucket"] = pd.cut(df["ret_lev_pct"], bins=ret_bins, labels=ret_labels)

    # entry_score 强度分桶
    if "entry_score" in df.columns:
        df["abs_entry_score"] = df["entry_score"].abs()
        score_bins = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
        score_labels = ["<=0.5", "0.5~1.0", "1.0~1.5", "1.5~2.0", ">2.0"]
        df["entry_score_bucket"] = pd.cut(df["abs_entry_score"], bins=score_bins, labels=score_labels)

    # 退出时 alpha 强度
    if "aligned_score_at_exit" in df.columns:
        df["abs_exit_aligned_score"] = df["aligned_score_at_exit"].abs()

    return df


def print_overview(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("总体表现")
    print("=" * 80)

    total_trades = len(df)
    total_net_pnl = df["net_pnl"].sum()
    total_fee = df["fee_usdt"].sum()
    avg_net_pnl = df["net_pnl"].mean()
    wr = win_rate(df["net_pnl"])
    avg_hold = df["hold_seconds"].mean()
    avg_ret = df["ret_lev_pct"].mean()

    gross_profit = df.loc[df["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = df.loc[df["net_pnl"] < 0, "net_pnl"].sum()
    profit_factor = safe_div(gross_profit, abs(gross_loss)) if gross_loss != 0 else np.nan

    print(f"总交易数         : {total_trades}")
    print(f"总净PnL          : {total_net_pnl:.4f}")
    print(f"总手续费         : {total_fee:.4f}")
    print(f"平均每笔净PnL    : {avg_net_pnl:.4f}")
    print(f"胜率             : {wr:.2%}")
    print(f"平均杠杆收益率   : {avg_ret:.2f}%")
    print(f"平均持仓时间     : {avg_hold:.1f}s")
    print(f"Profit Factor    : {profit_factor:.3f}")

    print("\n短单 / 长单对比")
    short = df[df["hold_seconds"] < 60]
    mid = df[(df["hold_seconds"] >= 60) & (df["hold_seconds"] < 120)]
    long = df[df["hold_seconds"] >= 120]

    for name, sub in [("短单(<60s)", short), ("中单(60-120s)", mid), ("长单(>=120s)", long)]:
        if len(sub) == 0:
            continue
        print(
            f"{name:<14} "
            f"交易数={len(sub):>4}  "
            f"总PnL={sub['net_pnl'].sum():>8.4f}  "
            f"均PnL={sub['net_pnl'].mean():>8.4f}  "
            f"胜率={win_rate(sub['net_pnl']):>7.2%}"
        )


def print_top_bottom(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("最佳 / 最差交易")
    print("=" * 80)

    cols = ["trade_id", "symbol", "side", "exit_reason", "hold_seconds", "ret_lev_pct", "net_pnl"]
    best = df.sort_values("net_pnl", ascending=False).head(10)[cols]
    worst = df.sort_values("net_pnl", ascending=True).head(10)[cols]

    print("\nTop 10 Best Trades")
    print(best.to_string(index=False))

    print("\nTop 10 Worst Trades")
    print(worst.to_string(index=False))


def analyze_filters(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("过滤实验：如果过滤短持仓，会发生什么")
    print("=" * 80)

    thresholds = [30, 60, 90, 120, 180]
    rows = []
    for t in thresholds:
        sub = df[df["hold_seconds"] >= t]
        if len(sub) == 0:
            continue
        rows.append({
            "min_hold_seconds": t,
            "trades": len(sub),
            "total_pnl": sub["net_pnl"].sum(),
            "avg_pnl": sub["net_pnl"].mean(),
            "win_rate": win_rate(sub["net_pnl"]),
            "avg_ret_lev_pct": sub["ret_lev_pct"].mean(),
        })

    res = pd.DataFrame(rows)
    if not res.empty:
        print(res.to_string(index=False))
    return res


def analyze_entry_score(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("entry_score 强度分析")
    print("=" * 80)

    if "entry_score_bucket" not in df.columns:
        print("缺少 entry_score，跳过。")
        return pd.DataFrame()

    grouped = summarize_group(df.dropna(subset=["entry_score_bucket"]), "entry_score_bucket")
    print(grouped.to_string(index=False))
    return grouped


def analyze_exit_reason(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("按 exit_reason 分析")
    print("=" * 80)

    grouped = summarize_group(df, "exit_reason")
    print(grouped.to_string(index=False))
    return grouped


def analyze_symbol(df: pd.DataFrame, min_trades: int = 3):
    print("\n" + "=" * 80)
    print("按 symbol 分析")
    print("=" * 80)

    grouped = summarize_group(df, "symbol")
    grouped = grouped[grouped["trades"] >= min_trades].reset_index(drop=True)

    print("\n表现最好（至少 3 笔）")
    print(grouped.sort_values("total_pnl", ascending=False).head(15).to_string(index=False))

    print("\n表现最差（至少 3 笔）")
    print(grouped.sort_values("total_pnl", ascending=True).head(15).to_string(index=False))

    return grouped


def analyze_hold_bucket(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("按持仓时间分桶分析")
    print("=" * 80)

    grouped = summarize_group(df.dropna(subset=["hold_bucket"]), "hold_bucket")
    print(grouped.to_string(index=False))
    return grouped


def analyze_state(df: pd.DataFrame):
    if "lc_state_at_exit" not in df.columns:
        print("\n缺少 lc_state_at_exit，跳过状态分析。")
        return pd.DataFrame()

    print("\n" + "=" * 80)
    print("按生命周期状态分析")
    print("=" * 80)

    grouped = summarize_group(df, "lc_state_at_exit")
    print(grouped.to_string(index=False))
    return grouped


def save_outputs(out_dir: Path, outputs: dict[str, pd.DataFrame]):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        if isinstance(data, pd.DataFrame) and not data.empty:
            data.to_csv(out_dir / f"{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze trades.csv from Alpha Factory.")
    parser.add_argument("--file", required=True, help="Path to trades.csv")
    parser.add_argument("--out", default="trade_analysis_output", help="Output directory")
    args = parser.parse_args()

    file_path = Path(args.file)
    out_dir = Path(args.out)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = [
        "trade_id", "symbol", "side", "entry_score", "exit_reason",
        "hold_seconds", "ret_lev_pct", "fee_usdt", "net_pnl"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = add_buckets(df)

    print_overview(df)
    print_top_bottom(df)

    exit_reason_df = analyze_exit_reason(df)
    hold_bucket_df = analyze_hold_bucket(df)
    state_df = analyze_state(df)
    symbol_df = analyze_symbol(df)
    entry_score_df = analyze_entry_score(df)
    hold_filter_df = analyze_filters(df)

    outputs = {
        "by_exit_reason": exit_reason_df,
        "by_hold_bucket": hold_bucket_df,
        "by_lifecycle_state": state_df,
        "by_symbol": symbol_df,
        "by_entry_score_bucket": entry_score_df,
        "hold_filter_experiment": hold_filter_df,
    }
    save_outputs(out_dir, outputs)

    print("\n" + "=" * 80)
    print(f"分析结果已保存到: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
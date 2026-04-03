"""
Price Trail Viewer - 持仓价格轨迹可视化

用法：
    python view_price_trail.py                   # 读取默认目录 ./price_trails/，显示最近20笔
    python view_price_trail.py --dir /path/to    # 指定目录
    python view_price_trail.py --n 50            # 显示最近50笔
    python view_price_trail.py --symbol BTCUSDT  # 只看某品种
    python view_price_trail.py --live            # 每30秒自动刷新

图表说明：
    纵轴 : 保证金收益率 %（= 价格变动% × 杠杆）
    横轴 : 持仓时间（秒）
    颜色 : 绿色=盈利平仓，红色=止损，橙色=强平，蓝色=其他
    虚线 : SL 和 TP 水平线
    ×标记: 平仓点
"""

import argparse
import csv
import os
import time
from pathlib import Path


def parse_trail_file(fpath: str) -> dict | None:
    """解析单个轨迹 CSV 文件，返回结构化数据"""
    try:
        with open(fpath, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 3:
            return None

        # 解析元数据行（第一行）
        meta = {}
        for cell in rows[0]:
            cell = cell.strip().lstrip("# ")
            if "=" in cell:
                k, v = cell.split("=", 1)
                meta[k.strip()] = v.strip()

        entry_price = float(meta.get("entry_price", 0))
        exit_price  = float(meta.get("exit_price",  0))
        sl_price    = float(meta.get("sl_price",    0)) if meta.get("sl_price", "None") != "None" else None
        tp_price    = float(meta.get("tp_price",    0)) if meta.get("tp_price", "None") != "None" else None
        lev         = float(meta.get("leverage",    1))
        side        = meta.get("side", "LONG")
        symbol      = meta.get("symbol", Path(fpath).stem)
        reason      = meta.get("exit_reason", "unknown")

        # 解析数据行（跳过第1行元数据、第2行表头）
        points = []
        for row in rows[2:]:
            if len(row) < 4:
                continue
            try:
                ts      = float(row[0])
                secs    = float(row[1])
                price   = float(row[2])
                ret_m   = float(row[3])
                points.append((secs, price, ret_m))
            except (ValueError, IndexError):
                continue

        if not points:
            return None

        # SL/TP 转为保证金收益率
        def to_margin_ret(p):
            if p is None or entry_price <= 0:
                return None
            if side == "LONG":
                return (p - entry_price) / entry_price * lev * 100
            else:
                return (entry_price - p) / entry_price * lev * 100

        return {
            "symbol":       symbol,
            "side":         side,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "exit_reason":  reason,
            "sl_ret":       to_margin_ret(sl_price),
            "tp_ret":       to_margin_ret(tp_price),
            "leverage":     lev,
            "points":       points,   # [(secs, price, ret_margin_pct)]
            "final_ret":    points[-1][2],
            "hold_secs":    points[-1][0],
            "mtime":        os.path.getmtime(fpath),
        }
    except Exception as e:
        print(f"[warn] 解析失败 {fpath}: {e}")
        return None


def load_trails(trail_dir: str, symbol_filter: str = None, n: int = 20) -> list:
    """加载最近 n 笔轨迹"""
    trail_dir = Path(trail_dir)
    if not trail_dir.exists():
        print(f"[error] 目录不存在: {trail_dir}")
        return []

    files = sorted(trail_dir.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)

    if symbol_filter:
        files = [f for f in files if symbol_filter.upper() in f.name.upper()]

    files = files[:n]

    trails = []
    for f in files:
        t = parse_trail_file(str(f))
        if t:
            trails.append(t)

    return trails


def plot_trails(trails: list, title: str = "Price Trail"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not trails:
        print("无可显示的轨迹数据")
        return

    # 颜色规则
    def trail_color(t):
        r = t["exit_reason"]
        ret = t["final_ret"]
        if "take_profit" in r or "profit_lock" in r:
            return "#00c853"   # 绿色
        if "stop_loss" in r or "max_loss" in r:
            return "#f44336"   # 红色
        if ret > 0:
            return "#66bb6a"   # 淡绿
        return "#ef9a9a"       # 淡红

    n = len(trails)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, trail in enumerate(trails):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[0][c]

        secs  = [p[0] for p in trail["points"]]
        rets  = [p[2] for p in trail["points"]]
        color = trail_color(trail)

        ax.plot(secs, rets, color=color, linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        # SL / TP 水平线
        if trail["sl_ret"] is not None:
            ax.axhline(trail["sl_ret"], color="#f44336", linewidth=1.0,
                       linestyle=":", alpha=0.8, label=f"SL {trail['sl_ret']:.1f}%")
        if trail["tp_ret"] is not None:
            ax.axhline(trail["tp_ret"], color="#00c853", linewidth=1.0,
                       linestyle=":", alpha=0.8, label=f"TP {trail['tp_ret']:.1f}%")

        # 平仓点标记
        ax.plot(secs[-1], rets[-1], "x", color=color, markersize=8, markeredgewidth=2)

        hold_str = f"{trail['hold_secs']:.0f}s" if trail["hold_secs"] < 3600 else f"{trail['hold_secs']/3600:.1f}h"
        ax.set_title(
            f"{trail['symbol']} {trail['side']}\n"
            f"{trail['exit_reason']}  {trail['final_ret']:+.1f}%  {hold_str}",
            fontsize=8,
            color=color,
        )
        ax.set_xlabel("持仓时间(秒)", fontsize=7)
        ax.set_ylabel("保证金收益%", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)

    # 隐藏多余格子
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[0][c]
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="持仓价格轨迹查看器")
    parser.add_argument("--dir",    default="./price_trails", help="轨迹文件目录")
    parser.add_argument("--n",      type=int, default=20,     help="显示最近N笔")
    parser.add_argument("--symbol", default=None,             help="过滤品种")
    parser.add_argument("--live",   action="store_true",      help="每30秒自动刷新")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("TkAgg")  # Windows 下使用 TkAgg backend

    if args.live:
        print(f"[Live模式] 每30秒刷新，目录={args.dir}，Ctrl+C 退出")
        while True:
            trails = load_trails(args.dir, args.symbol, args.n)
            plot_trails(trails, title=f"Price Trails ({len(trails)} 笔) — {time.strftime('%H:%M:%S')}")
            time.sleep(30)
    else:
        trails = load_trails(args.dir, args.symbol, args.n)
        print(f"加载到 {len(trails)} 笔轨迹")
        for t in trails:
            print(f"  {t['symbol']:12s} {t['side']:5s} {t['exit_reason']:20s} {t['final_ret']:+6.1f}%  {t['hold_secs']:.0f}s")
        plot_trails(trails, title=f"Price Trails — 最近{len(trails)}笔")


if __name__ == "__main__":
    main()

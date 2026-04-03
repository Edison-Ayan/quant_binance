"""
监控引擎（MonitorEngine）- 终极版

定时打印：
    原有：持仓 qty / entry / unrealized_pnl
    新增：unified alpha / lifecycle 状态 / market state / 成本过滤 / 风控拒绝统计

依然使用 threading.Timer 自调度模式（理由见原版注释）。

可选注入策略引用（strategy）：
    - 若传入 AlphaFactoryStrategy，则可显示 unified alpha、lifecycle、market state
    - 不传入时退化为原版只显示 PositionManager 数据（向后兼容）
"""

import threading
import time


class MonitorEngine:
    """
    持仓状态定时监控引擎（终极版）。

    新参数：
        strategy (optional) : AlphaFactoryStrategy 实例，提供高层状态
        report_engine (optional) : ReportEngine 实例，提供过滤/拒绝统计
    """

    def __init__(
        self,
        position_manager,
        interval: int = 30,
        strategy=None,
        report_engine=None,
    ):
        self.position_manager = position_manager
        self.interval         = interval
        self.strategy         = strategy        # 可选注入
        self.report_engine    = report_engine   # 可选注入
        self._timer: threading.Timer = None

    # ─── 生命周期 ─────────────────────────────────────────────────────────────

    def start(self):
        self._schedule()

    def stop(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    # ─── 自调度 ───────────────────────────────────────────────────────────────

    def _schedule(self):
        self.print_status()
        self._timer = threading.Timer(self.interval, self._schedule)
        self._timer.daemon = True
        self._timer.start()

    # ─── 状态打印 ─────────────────────────────────────────────────────────────

    def print_status(self):
        sep = "-" * 62
        print(sep)
        print(f"[Monitor] {time.strftime('%H:%M:%S')}  持仓快照")
        print(sep)

        # ── 1. 持仓状态（PositionManager）────────────────────────────────────
        positions = self.position_manager.positions
        active    = {s: p for s, p in positions.items() if p.qty != 0}

        if not active:
            print("  无持仓")
        else:
            for symbol, pos in active.items():
                # 从 strategy 获取 lifecycle 和 unified alpha（如果有）
                lc_info     = ""
                alpha_info  = ""
                if self.strategy:
                    lc = self.strategy.lifecycle_tracker.get_lifecycle(symbol, "LONG") \
                         or self.strategy.lifecycle_tracker.get_lifecycle(symbol, "SHORT")
                    if lc:
                        lc_info = f"  [{lc.state.value:<10}]"
                    fa = self.strategy._latest_fused_alphas.get(symbol)
                    if fa:
                        alpha_info = f"  α={fa.unified:+.3f}"

                print(
                    f"  {symbol:<12} "
                    f"qty={pos.qty:>10.4f}  "
                    f"entry={pos.entry_price:>12.4f}  "
                    f"pnl={pos.unrealized_pnl:>+10.4f}"
                    f"{lc_info}{alpha_info}"
                )

        # ── 2. 市场状态（MarketStateEngine）──────────────────────────────────
        if self.strategy and self.strategy._latest_market_state:
            ms = self.strategy._latest_market_state
            print(sep)
            print(
                f"  [市场状态] "
                f"regime={ms.regime.value:<16} "
                f"tradability={ms.tradability:.3f}  "
                f"dispersion={ms.dispersion:.5f}  "
                f"crowding_z={ms.crowding_score:+.2f}  "
                f"tradeable={'✓' if ms.is_tradeable else '✗'}"
            )

        # ── 3. 候选池摘要 ─────────────────────────────────────────────────────
        if self.strategy and hasattr(self.strategy, "_candidate_pool"):
            pool  = self.strategy._candidate_pool
            longs = list(pool.get("long",  set()))[:5]
            shorts= list(pool.get("short", set()))[:5]
            print(
                f"  [候选池]   "
                f"多头={longs}  "
                f"空头={shorts}"
            )

        # ── 4. Lifecycle 分布（开放仓位）────────────────────────────────────
        if self.strategy:
            lc_status = self.strategy.lifecycle_tracker.get_all_status()
            if lc_status:
                from collections import Counter
                state_dist = Counter(v["state"] for v in lc_status.values())
                print(
                    f"  [生命周期] "
                    + "  ".join(f"{s}:{n}" for s, n in state_dist.items())
                )

        # ── 5. 成本过滤 / 风控统计（ReportEngine）────────────────────────────
        if self.report_engine:
            cost_n = len(self.report_engine._cost_rejects)
            risk_n = len(self.report_engine._risk_rejects)
            print(f"  [过滤统计] 成本拒绝={cost_n}次  风控拒绝={risk_n}次")

        print(sep)

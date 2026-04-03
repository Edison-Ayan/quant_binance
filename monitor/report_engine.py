"""
ReportEngine - Session 级别完整运行报告

在程序停止时生成一份覆盖整个 session 的汇总报告，回答：

    策略层：
        做了多少笔交易？各退出原因占比？
        生命周期分布（BUILD/EXPANSION/DECAY/REVERSAL 各占多少）？
        持仓时长分布？

    市场状态层：
        各 regime 下各做了多少笔交易？表现如何？
        平均 tradability？平均 dispersion？

    成本过滤层：
        有多少信号被 CostModel 拒绝？节省了多少潜在亏损？

    风控层：
        各风控门的拒绝次数？

    PnL 归因：
        总毛盈亏、总手续费、净盈亏
        分因子贡献（Top 3 赚钱因子 / Top 3 亏钱因子）
        胜率、盈亏比、平均持仓时长

调用方式：
    engine = ReportEngine()

    # 在运行期间收集事件
    engine.record_cost_reject(symbol, reason)
    engine.record_risk_reject(symbol, reason)
    engine.record_lifecycle_exit(symbol, side, state)

    # 在 stop() 时调用
    engine.generate(strategy=..., logger=logger)
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReportSnapshot:
    """report_engine.generate() 的返回值，结构化报告"""
    session_start:     float = 0.0
    session_end:       float = 0.0
    session_seconds:   float = 0.0

    # 交易摘要
    total_trades:      int   = 0
    win_trades:        int   = 0
    loss_trades:       int   = 0
    win_rate:          float = 0.0
    avg_hold_s:        float = 0.0
    total_pnl_usdt:    float = 0.0
    total_fee_usdt:    float = 0.0
    net_pnl_usdt:      float = 0.0
    avg_profit_usdt:   float = 0.0
    avg_loss_usdt:     float = 0.0
    profit_factor:     float = 0.0   # avg_profit / |avg_loss|

    # 退出原因分布
    exit_reasons:      dict  = field(default_factory=dict)

    # 生命周期退出分布
    lifecycle_exits:   dict  = field(default_factory=dict)

    # 市场状态分布
    regime_trade_count: dict = field(default_factory=dict)
    regime_pnl:         dict = field(default_factory=dict)
    avg_tradability:    float = 0.0
    avg_dispersion:     float = 0.0

    # 成本过滤统计
    cost_rejects:      int   = 0
    cost_reject_top:   dict  = field(default_factory=dict)

    # 风控统计
    risk_rejects:      int   = 0
    risk_reject_by_layer: dict = field(default_factory=dict)

    # 因子归因
    factor_pnl:        dict  = field(default_factory=dict)   # {factor_name: pnl_sum}


class ReportEngine:
    """
    运行报告引擎。

    采集 session 期间的统计数据，在程序结束时生成结构化报告。
    所有收集方法均为 O(1)，不影响主线程性能。
    """

    def __init__(self):
        self._start_time    = time.time()
        self._cost_rejects: List[dict]   = []
        self._risk_rejects: List[dict]   = []
        self._lifecycle_exits: List[dict] = []
        self._regime_log:   List[dict]   = []   # market state snapshots

    # ─── 事件收集接口（主线程调用）────────────────────────────────────────────

    def record_cost_reject(self, symbol: str, side: str, reason: str):
        self._cost_rejects.append({"symbol": symbol, "side": side, "reason": reason})

    def record_risk_reject(self, symbol: str, layer: str, reason: str = ""):
        self._risk_rejects.append({"symbol": symbol, "layer": layer, "reason": reason})

    def record_lifecycle_exit(self, symbol: str, side: str, state: str):
        self._lifecycle_exits.append({"symbol": symbol, "side": side, "state": state})

    def record_market_state(self, regime: str, tradability: float, dispersion: float):
        self._regime_log.append({
            "regime":      regime,
            "tradability": tradability,
            "dispersion":  dispersion,
        })

    # ─── 报告生成 ─────────────────────────────────────────────────────────────

    def generate(self, strategy=None, logger=None) -> ReportSnapshot:
        """
        生成并打印完整的 session 报告。

        参数：
            strategy : AlphaFactoryStrategy 实例（可选，用于读取交易历史和因子）
            logger   : logger 实例（可选，不传则用 print）
        """
        now  = time.time()
        snap = ReportSnapshot(
            session_start   = self._start_time,
            session_end     = now,
            session_seconds = now - self._start_time,
        )

        # ── 从 strategy 读取交易历史 ─────────────────────────────────────────
        trades = []
        if strategy and hasattr(strategy, "_trades"):
            trades = list(strategy._trades)

        if trades:
            pnl_list   = [t.pnl_usdt - t.fee_usdt for t in trades]  # 净pnl
            gross_list = [t.pnl_usdt for t in trades]
            fee_list   = [t.fee_usdt for t in trades]
            win_list   = [p for p in pnl_list if p > 0]
            loss_list  = [p for p in pnl_list if p <= 0]

            snap.total_trades    = len(trades)
            snap.win_trades      = len(win_list)
            snap.loss_trades     = len(loss_list)
            snap.win_rate        = snap.win_trades / snap.total_trades if snap.total_trades else 0.0
            snap.total_pnl_usdt  = sum(gross_list)
            snap.total_fee_usdt  = sum(fee_list)
            snap.net_pnl_usdt    = sum(pnl_list)
            snap.avg_hold_s      = sum(t.hold_seconds for t in trades) / len(trades)
            snap.avg_profit_usdt = sum(win_list)  / len(win_list)  if win_list  else 0.0
            snap.avg_loss_usdt   = sum(loss_list) / len(loss_list) if loss_list else 0.0
            if snap.avg_loss_usdt < 0:
                snap.profit_factor = abs(snap.avg_profit_usdt / snap.avg_loss_usdt)

            snap.exit_reasons  = dict(Counter(t.reason for t in trades))
            snap.factor_pnl    = self._calc_factor_pnl(trades)

            # regime 归因（从记录的市场状态快照对应）
            regime_counts: Counter = Counter(t.reason for t in trades
                                             if hasattr(t, "regime"))

        # ── 生命周期退出分布 ─────────────────────────────────────────────────
        snap.lifecycle_exits = dict(Counter(e["state"] for e in self._lifecycle_exits))

        # ── 市场状态统计 ─────────────────────────────────────────────────────
        if self._regime_log:
            snap.avg_tradability = sum(r["tradability"] for r in self._regime_log) / len(self._regime_log)
            snap.avg_dispersion  = sum(r["dispersion"]  for r in self._regime_log) / len(self._regime_log)
            snap.regime_trade_count = dict(Counter(r["regime"] for r in self._regime_log))

        # ── 成本过滤统计 ─────────────────────────────────────────────────────
        snap.cost_rejects   = len(self._cost_rejects)
        snap.cost_reject_top = dict(Counter(r["reason"].split(":")[0] for r in self._cost_rejects).most_common(5))

        # ── 风控统计 ─────────────────────────────────────────────────────────
        snap.risk_rejects          = len(self._risk_rejects)
        snap.risk_reject_by_layer  = dict(Counter(r["layer"] for r in self._risk_rejects))

        # ── 打印报告 ─────────────────────────────────────────────────────────
        self._print(snap, logger)
        return snap

    # ─── 打印 ─────────────────────────────────────────────────────────────────

    def _print(self, snap: ReportSnapshot, logger):
        _log = logger.info if logger else print
        sep  = "=" * 62

        _log(sep)
        _log("  Alpha Factory - Session Report")
        _log(sep)
        _log(f"  运行时长  : {snap.session_seconds/60:.1f} 分钟")
        _log("")

        _log("── 交易摘要 ───────────────────────────────────────────────")
        _log(f"  总交易笔数: {snap.total_trades}")
        _log(f"  胜率      : {snap.win_rate*100:.1f}%  "
             f"(赢{snap.win_trades} / 亏{snap.loss_trades})")
        _log(f"  总毛盈亏  : {snap.total_pnl_usdt:+.3f} USDT")
        _log(f"  总手续费  : -{snap.total_fee_usdt:.3f} USDT")
        _log(f"  净盈亏    : {snap.net_pnl_usdt:+.3f} USDT")
        _log(f"  盈亏比    : {snap.profit_factor:.2f}x")
        _log(f"  均持仓时长: {snap.avg_hold_s:.0f}s")
        _log(f"  平均盈利  : {snap.avg_profit_usdt:+.4f} USDT")
        _log(f"  平均亏损  : {snap.avg_loss_usdt:+.4f} USDT")

        _log("")
        _log("── 退出原因分布 ────────────────────────────────────────────")
        for reason, cnt in sorted(snap.exit_reasons.items(), key=lambda x: -x[1]):
            pct = cnt / snap.total_trades * 100 if snap.total_trades else 0
            _log(f"  {reason:<30} {cnt:>4}笔  {pct:>5.1f}%")

        _log("")
        _log("── Alpha 生命周期退出分布 ──────────────────────────────────")
        if snap.lifecycle_exits:
            for state, cnt in snap.lifecycle_exits.items():
                _log(f"  {state:<20} {cnt:>4}笔")
        else:
            _log("  (无生命周期退出记录)")

        _log("")
        _log("── 市场状态 ────────────────────────────────────────────────")
        _log(f"  平均可交易性: {snap.avg_tradability:.3f}")
        _log(f"  平均分散度  : {snap.avg_dispersion:.5f}")
        if snap.regime_trade_count:
            _log("  Regime 分布:")
            for regime, cnt in snap.regime_trade_count.items():
                _log(f"    {regime:<20} {cnt:>4}次")

        _log("")
        _log("── 成本过滤 ────────────────────────────────────────────────")
        _log(f"  成本拒绝次数: {snap.cost_rejects}")
        if snap.cost_reject_top:
            for reason, cnt in snap.cost_reject_top.items():
                _log(f"    {reason:<30} {cnt:>4}次")

        _log("")
        _log("── 风控统计 ────────────────────────────────────────────────")
        _log(f"  风控拒绝总数: {snap.risk_rejects}")
        if snap.risk_reject_by_layer:
            for layer, cnt in snap.risk_reject_by_layer.items():
                _log(f"    {layer:<30} {cnt:>4}次")

        _log("")
        _log("── 因子归因（Top 5）────────────────────────────────────────")
        if snap.factor_pnl:
            for fname, pnl in sorted(snap.factor_pnl.items(), key=lambda x: -abs(x[1]))[:5]:
                _log(f"  {fname:<30} {pnl:+.4f}")
        else:
            _log("  (无因子数据)")

        _log(sep)

    # ─── 工具 ─────────────────────────────────────────────────────────────────

    def _calc_factor_pnl(self, trades: list) -> dict:
        """
        计算每个因子的净 PnL 贡献（因子值 × 净收益率，跨所有交易求和）。
        仅作方向性归因参考，非严格 IC。
        """
        factor_pnl: dict = defaultdict(float)
        factor_cnt: dict = defaultdict(int)
        for t in trades:
            ret_lev = t.ret_lev_pct  # 杠杆后保证金收益率（已标准化，无量纲）
            for fname, fval in (t.factors or {}).items():
                if fname == "total" or fname.endswith("_contrib"):
                    continue
                # LONG 时因子值正=利多，SHORT 时反向
                aligned_fval = fval if t.side == "LONG" else -fval
                factor_pnl[fname] += aligned_fval * ret_lev  # 用收益率而非USDT绝对值
                factor_cnt[fname] += 1
        # 除以交易数得到平均贡献，消除样本量影响
        return {k: v / factor_cnt[k] for k, v in factor_pnl.items() if factor_cnt[k] > 0}

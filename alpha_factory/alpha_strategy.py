"""
Alpha Factory Strategy - 全市场多空双向策略

双层架构：
    慢层（ScoringEngine）：每 rank_interval 秒运行，选出候选池（Top 15 / Bottom 15）
    快层（LOBTimingEngine）：on_tick 实时运行，在候选池内判断微观结构进场时机

多头逻辑：慢层选出候选 → 快层 microprice+OFI+lob_z2 timing_score 超阈值 → 开多
空头逻辑：慢层选出候选 → 快层 timing_score 反向超阈值 → 开空

仓位结构：
    self.long_positions  = {symbol: {entry_price, qty, entry_time, score}}
    self.short_positions = {symbol: {entry_price, qty, entry_time, score}}

退出机制：
    stop_loss     : 交易所侧条件单（硬底线）
    trailing_stop : on_tick 实时追踪（锁利润）
    take_profit   : 价格型止盈（快速锁利）
    alpha_flip    : LOBTimingEngine timing_score 反向超 EXIT_THRESHOLD

同品种互斥：
    不允许同时持有同一品种的多空仓位，进场时检查对立仓位。
"""

import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List

from strategy.strategy_base import StrategyBase
from data_layer.logger import logger

from .feature_engine import FeatureEngine
from .scoring_engine import ScoringEngine
from .ranking_engine import RankingEngine
from .lob_manifold_engine import LOBManifoldEngine
from .shock_detector import ShockDetector
from .lob_timing_engine import LOBTimingEngine


@dataclass
class TradeRecord:
    symbol:         str
    side:           str       # "LONG" / "SHORT"
    entry_price:    float
    exit_price:     float
    qty:            float
    entry_time:     float     # unix timestamp
    exit_time:      float
    reason:         str       # take_profit / stop_loss / strategy_stop / side_switch
    leverage:       int
    pnl_usdt:       float     # qty * |price_change|，正=盈利
    ret_pct:        float     # 价格变动 %
    ret_lev_pct:    float     # 杠杆后保证金收益率 %
    hold_seconds:   float
    fee_usdt:       float = 0.0               # 手续费（开仓+平仓各一次 taker fee）
    factors:        dict = field(default_factory=dict)  # 开仓时各因子得分贡献


class AlphaFactoryStrategy(StrategyBase):

    DEFAULT_PARAMS = {
        "rank_interval":       60,     # 重排序间隔（秒）
        "max_long_positions":   3,     # 最大同时做多品种数
        "max_short_positions":  3,     # 最大同时做空品种数
        "long_score_threshold":  0.5,  # 候选池做多最低 EMA 分数（慢层门槛）
        "short_score_threshold":-0.5,  # 候选池做空最高 EMA 分数（慢层门槛）
        "ema_alpha":             0.4,  # EMA 平滑系数
        "confirm_rounds":        2,    # 连续上榜 N 次才进入候选池
        "ranking_top_n":        15,    # 候选池大小（Top 15 多 / Bottom 15 空）
        "leverage":             10,    # 杠杆倍数（与交易所设置保持一致）
        "stop_loss_pct":        0.008, # 价格下跌 0.8%  触发止损（与实际杠杆无关）
        "take_profit_pct":      0.015, # 价格上涨 1.5%  触发止盈（与实际杠杆无关）
        "trade_size_usdt":     100.0,  # 每笔下单金额（USDT）
        "min_volume_zscore":     0.3,  # 进场最低成交量 Z-score
        "warmup_count":          4,    # 热身期排序次数
        "max_spread_bps":       20.0,  # 最大可接受价差（基点）
        # ── 手续费 ──────────────────────────────────────────────────────────
        "fee_rate":             0.0004, # Binance taker fee（0.04%），开平各一次
        # ── 动态仓位 ─────────────────────────────────────────────────────────
        "size_min_scale":        0.7,  # 信号最弱时的仓位比例（score≈0 → 70% base_size）
        "size_max_scale":        1.5,  # 信号最强时的仓位比例（score≈3σ → 150% base_size）
        # ── LOB Timing Engine（快层：实时进场时机）────────────────────────────
        "timing_entry_threshold": 0.30,  # timing_score 超此值 → 触发开仓（microprice 单独强即可）
        "timing_exit_threshold":  0.15,  # timing_score 反向超此值 → alpha_flip 平仓（死区 [-0.15, +0.30]）
        # ── Regime filter ────────────────────────────────────────────────────
        "regime_btc_symbol":  "BTCUSDT",  # 用于判断市场制度的参考品种
        "regime_5m_threshold": 0.02,  # BTC 5分钟涨跌超此值（2%）→ 触发方向性过滤
        # ── 市场可交易性过滤 ─────────────────────────────────────────────────
        "min_market_activity": 0.3,   # 全市场 vol_zscore 绝对均值最小值（低于此=市场静止）
        "min_market_move":     0.001, # 全市场 5m 平均涨跌幅最小值（低于此=横盘磨损区）
        # ── 市场中性敞口控制 ─────────────────────────────────────────────────
        "max_net_exposure":  300.0,   # 多空名义价值最大净差额（USDT），超出则不再加方向
        # ── 止损（硬底线，防爆仓）────────────────────────────────────────────
        "sl_vol_mult":         3.0,   # SL = 近期波动率 × 此倍数（仅作价格保护底线）
        "vol_window":           20,   # 计算波动率用的价格序列长度（笔数）
        # ── 单笔最大亏损保护（防止极端亏损）──────────────────────────────────
        "max_single_loss_margin_pct": 0.10,  # 10% 保证金亏损即强制平仓
        # ── Trailing Stop（锁利润）────────────────────────────────────────────
        "trailing_vol_mult":   2.0,   # 回撤阈值 = vol × 此倍数（从最高/低价回撤触发）
        "trailing_min_profit": 0.006, # 只有盈利超过此比例才激活 trailing
        # ── Dispersion Filter（截面分散度过滤）───────────────────────────────
        "min_score_dispersion": 0.3,  # scores 标准差低于此值 → 市场无 alpha，跳过候选池更新
        # ── 最短持仓时间（防止信号噪声导致过早平仓）───────────────────────────
        "min_hold_seconds":        120,   # 最短持仓时间（秒）
        # ── 反向开仓冷却期（防止频繁 flip）─────────────────────────────────────
        "cooldown_seconds":        120,   # 平仓后反向冷却时间（秒），防止连续反向摩擦
        # ── 因子自动进化 ──────────────────────────────────────────────────────
        "evolve_interval":     10,    # 每隔 N 轮排序做一次权重进化
        "evolve_min_trades":   15,    # 触发进化所需最少近期交易笔数
        "evolve_buffer":       50,    # 近期交易滚动缓冲区大小
        # ── 组合优化 ──────────────────────────────────────────────────────────────
        "port_weight_scale":   0.4,   # 历史 Sharpe 对仓位的影响系数（0=关闭，1=最强）
        "port_min_trades":     5,     # 启用组合权重所需最少历史交易笔数
        "port_stats_window":   30,    # 每个品种滚动统计的最大历史笔数
        # ── 相关性去重 ────────────────────────────────────────────────────────────
        "max_corr_threshold":  0.85,  # 候选品种间相关性超此值则去重（保留排名更高的）
        "corr_history_len":    30,    # 计算相关性使用的 ret_1m 历史长度
        # ── 数据导出 ──────────────────────────────────────────────────────────────
        "export_data":         True,  # 是否导出 CSV/JSON 数据供 Dashboard 使用
        "export_dir":          ".",   # 数据文件导出目录（默认当前目录）
    }

    def __init__(self, engine, symbols: list, params: dict = None, db=None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(engine, "AlphaFactory", symbols, merged)
        self._db = db  # 可选：storage.database.Database 实例，传入后自动持久化交易记录

        self.feature_engine  = FeatureEngine()
        self.scoring_engine  = ScoringEngine()
        self.shock_detector  = ShockDetector()
        self.lob_engine      = LOBManifoldEngine()
        self.timing_engine   = LOBTimingEngine(
            entry_threshold = merged.get("timing_entry_threshold", 0.40),
            exit_threshold  = merged.get("timing_exit_threshold",  0.20),
        )
        self.ranking_engine  = RankingEngine(
            top_n                 = merged.get("ranking_top_n", 15),
            bottom_n              = merged.get("ranking_top_n", 15),
            long_score_threshold  = self.params["long_score_threshold"],
            short_score_threshold = self.params["short_score_threshold"],
            ema_alpha             = self.params["ema_alpha"],
            confirm_rounds        = self.params["confirm_rounds"],
        )

        # 候选池：慢层每轮排序后更新，快层 on_tick 实时读取
        self._candidate_pool: Dict[str, set] = {"long": set(), "short": set()}
        # 最新特征快照（每轮排序时更新，供 on_tick 进场质量检查使用）
        self._latest_features: dict = {}
        # 最新得分快照（供动态仓位计算使用）
        self._latest_scores: dict = {}

        # 多空仓位分开管理
        self.long_positions:  Dict[str, dict] = {}
        self.short_positions: Dict[str, dict] = {}

        self._trades: List[TradeRecord] = []
        self._recent_trades: deque = deque(maxlen=self.params["evolve_buffer"])
        self._last_rank_time: float = 0.0
        self._tick_count: int = 0

        # 组合优化：每个品种的历史收益率（用于 Sharpe proxy）
        self._symbol_stats:    Dict[str, deque] = {}
        # 相关性去重：每个品种的 ret_1m 快照历史
        self._ret1m_history:   Dict[str, deque] = {}

        # 数据导出：初始化文件路径
        _dir = self.params["export_dir"]
        self._factor_log_path    = os.path.join(_dir, "factor_log.csv")
        self._ic_log_path        = os.path.join(_dir, "ic_log.csv")
        self._positions_json_path = os.path.join(_dir, "positions_current.json")
        # 追踪每个确认品种连续质量门槛失败次数；超过阈值后从确认池驱逐，腾出位置
        self._qgate_fails: Dict[str, int] = {}
        # 平仓时间记录：{symbol: {"LONG": ts, "SHORT": ts}}，用于反向开仓冷却期计算
        self._recently_closed: Dict[str, dict] = {}
        # 交易所侧 TP/SL 单追踪 {symbol: {tp_id, sl_id, tp_price, sl_price}}
        self._tp_sl_orders: Dict[str, dict] = {}

        lev = self.params["leverage"]
        sl  = self.params["stop_loss_pct"] * 100
        tp  = self.params["take_profit_pct"] * 100
        fee = self.params["fee_rate"] * 100
        logger.info(
            f"[AlphaFactory] 初始化完成 | "
            f"多头 Top{self.params['max_long_positions']} "
            f"空头 Bottom{self.params['max_short_positions']} | "
            f"杠杆={lev}x | 止损={sl:.1f}% 止盈={tp:.1f}% | "
            f"手续费={fee:.3f}%×2 | "
            f"仓位区间=[{self.params['size_min_scale']:.0%}~{self.params['size_max_scale']:.0%}]×base"
        )

    # ─── 事件回调 ────────────────────────────────────────────────────────────

    def on_tick(self, event):
        d   = event.data
        sym = d["symbol"]

        price = d["price"]
        self.feature_engine.on_trade(
            symbol         = sym,
            price          = price,
            qty            = d["qty"],
            is_buyer_maker = d["is_buyer_maker"],
            ts_ms          = d["timestamp"],
        )
        self._tick_count += 1

        # ── 插针检测 L1（成交维度）────────────────────────────────────────────
        shock = self.shock_detector.on_trade(
            symbol   = sym,
            price    = price,
            usdt_vol = price * d["qty"],
            ts_ms    = d["timestamp"],
        )
        if shock.is_shocked and (sym in self.long_positions or sym in self.short_positions):
            logger.warning(
                f"[ShockDetector] {sym} 流动性冲击 {shock.reasons}，紧急平仓"
            )
            if sym in self.long_positions:
                self._close_long(sym, "shock_exit")
            if sym in self.short_positions:
                self._close_short(sym, "shock_exit")

        # 实时 trailing stop：每 tick 检查持仓最高/低价并判断回撤
        if sym in self.long_positions:
            pos = self.long_positions[sym]
            if price > pos.get("max_price", pos["entry_price"]):
                pos["max_price"] = price
            elif self.params["trailing_vol_mult"] > 0:
                entry = pos["entry_price"]
                ret   = (price - entry) / entry if entry > 0 else 0.0
                if ret > self.params["trailing_min_profit"]:
                    vol       = self._calc_volatility(sym)
                    threshold = vol * self.params["trailing_vol_mult"]
                    max_p     = pos.get("max_price", entry)
                    if max_p > 0 and (max_p - price) / max_p > threshold:
                        self._close_long(sym, "trailing_stop")

        elif sym in self.short_positions:
            pos = self.short_positions[sym]
            if price < pos.get("min_price", pos["entry_price"]):
                pos["min_price"] = price
            elif self.params["trailing_vol_mult"] > 0:
                entry = pos["entry_price"]
                ret   = (entry - price) / entry if entry > 0 else 0.0
                if ret > self.params["trailing_min_profit"]:
                    vol       = self._calc_volatility(sym)
                    threshold = vol * self.params["trailing_vol_mult"]
                    min_p     = pos.get("min_price", entry)
                    if min_p > 0 and (price - min_p) / min_p > threshold:
                        self._close_short(sym, "trailing_stop")

        # ── 价格型止盈（补充信号驱动平仓，确保利润不被全部回吐）────────────────
        # take_profit_pct=0 时关闭此逻辑
        _tp_pct = self.params["take_profit_pct"]
        if _tp_pct > 0:
            if sym in self.long_positions and sym not in self.short_positions:
                _entry = self.long_positions[sym]["entry_price"]
                if _entry > 0 and price >= _entry * (1 + _tp_pct):
                    self._close_long(sym, "take_profit")
            elif sym in self.short_positions and sym not in self.long_positions:
                _entry = self.short_positions[sym]["entry_price"]
                if _entry > 0 and price <= _entry * (1 - _tp_pct):
                    self._close_short(sym, "take_profit")

        # ── 单笔最大亏损保护（程序端硬止损，exchange-side SL 的最后保障）────────
        # 防止 SL 单未成交或极端行情导致保证金亏损超过阈值
        _max_loss = self.params.get("max_single_loss_margin_pct", 0.10)
        _lev      = self.params["leverage"]
        if sym in self.long_positions and sym not in self.short_positions:
            _entry = self.long_positions[sym]["entry_price"]
            if _entry > 0 and (price - _entry) / _entry * _lev < -_max_loss:
                logger.warning(
                    f"[AlphaFactory] {sym} LONG 触发最大亏损保护 "
                    f"margin_loss={(price-_entry)/_entry*_lev*100:.1f}% > {_max_loss*100:.0f}%"
                )
                self._close_long(sym, "max_loss_guard")
        elif sym in self.short_positions and sym not in self.long_positions:
            _entry = self.short_positions[sym]["entry_price"]
            if _entry > 0 and (_entry - price) / _entry * _lev < -_max_loss:
                logger.warning(
                    f"[AlphaFactory] {sym} SHORT 触发最大亏损保护 "
                    f"margin_loss={(_entry-price)/_entry*_lev*100:.1f}% > {_max_loss*100:.0f}%"
                )
                self._close_short(sym, "max_loss_guard")

        # ── LOB Timing：实时进场 + alpha_flip 平仓 ─────────────────────────────
        feat = self._latest_features.get(sym)
        if feat is not None:
            min_hold = self.params["min_hold_seconds"]
            now_t    = time.time()

            # alpha_flip 平仓（timing 反向超 EXIT_THRESHOLD）
            if sym in self.long_positions:
                held = now_t - self.long_positions[sym]["entry_time"]
                if held >= min_hold and self.timing_engine.should_exit_long(sym, feat):
                    self._close_long(sym, "alpha_flip")
            elif sym in self.short_positions:
                held = now_t - self.short_positions[sym]["entry_time"]
                if held >= min_hold and self.timing_engine.should_exit_short(sym, feat):
                    self._close_short(sym, "alpha_flip")

            # 实时开仓：候选池内 + timing_score 超阈值 + 未持仓 + 仓位未满
            if (sym in self._candidate_pool["long"]
                    and sym not in self.long_positions
                    and sym not in self.short_positions
                    and len(self.long_positions) < self.params["max_long_positions"]
                    and not self.shock_detector.is_kill_switched()
                    and self.timing_engine.should_enter_long(sym, feat)):
                regime = self._regime_filter(self._latest_features)
                if regime["allow_long"] and self._pass_quality_gate(sym, feat, "LONG"):
                    score = self._latest_scores.get(sym, 0.0)
                    self._open_long(sym, score, price, self._latest_features)
                    self.timing_engine.record_entry(sym)

            elif (sym in self._candidate_pool["short"]
                    and sym not in self.short_positions
                    and sym not in self.long_positions
                    and len(self.short_positions) < self.params["max_short_positions"]
                    and not self.shock_detector.is_kill_switched()
                    and self.timing_engine.should_enter_short(sym, feat)):
                regime = self._regime_filter(self._latest_features)
                if regime["allow_short"] and self._pass_quality_gate(sym, feat, "SHORT"):
                    score = self._latest_scores.get(sym, 0.0)
                    self._open_short(sym, score, price, self._latest_features)
                    self.timing_engine.record_entry(sym)

        now = time.time()
        if now - self._last_rank_time >= self.params["rank_interval"]:
            self._rank_and_trade(now)

    def on_order_book(self, event):
        d   = event.data
        sym = d.get("symbol", "")
        if not sym:
            return

        # 更新 OFI / spread / depth_imbalance（使用 best bid/ask）
        bid     = d.get("best_bid", 0.0)
        bid_qty = d.get("best_bid_qty", 0.0)
        ask     = d.get("best_ask", 0.0)
        ask_qty = d.get("best_ask_qty", 0.0)
        self.feature_engine.on_book_ticker(
            symbol  = sym,
            bid     = bid,
            bid_qty = bid_qty,
            ask     = ask,
            ask_qty = ask_qty,
        )
        # LOB Timing Engine：维护 microprice_delta 滚动归一化状态
        if bid > 0 and ask > 0 and bid_qty > 0 and ask_qty > 0:
            self.timing_engine.on_book_ticker(sym, bid, bid_qty, ask, ask_qty)

        # ── 插针检测 L1（盘口维度）────────────────────────────────────────────
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        spread_bps = (ask - bid) / mid * 10_000 if mid > 1e-10 else 0.0
        shock = self.shock_detector.on_book(
            symbol     = sym,
            spread_bps = spread_bps,
            bid_usdt   = bid * bid_qty,
            ask_usdt   = ask * ask_qty,
        )
        if shock.is_shocked and (sym in self.long_positions or sym in self.short_positions):
            logger.warning(
                f"[ShockDetector] {sym} 盘口冲击 {shock.reasons}，紧急平仓"
            )
            if sym in self.long_positions:
                self._close_long(sym, "shock_exit")
            if sym in self.short_positions:
                self._close_short(sym, "shock_exit")

        # LOB 路径已移至 on_lob_depth（由 @depth5@100ms 流驱动）

    def on_lob_depth(self, data: dict):
        """
        处理 @depth5@100ms 多档深度快照 → LOB 流形引擎。

        直接调用（不经过事件引擎），减少延迟。
        每个消息包含 top-5 bids/asks 完整快照，由 MultiSymbolFeed 解析后传入。
        """
        sym = data.get("symbol", "")
        mid = data.get("mid", 0.0)
        if not sym or mid <= 0:
            return

        self.lob_engine.on_order_book(
            symbol = sym,
            bids   = data.get("bids", []),
            asks   = data.get("asks", []),
            mid    = mid,
            ts     = data.get("ts_ms", 0) / 1000.0 or None,
        )
        z = self.lob_engine.get_symbol_latent(sym)
        if z is not None:
            bucket = self.lob_engine.get_symbol_bucket(sym)
            self.feature_engine.update_lob_latent(sym, z, bucket)

    def update_derivatives(self, symbol: str, funding_rate: float, oi: float):
        self.feature_engine.update_derivatives(symbol, funding_rate, oi)

    # ─── 核心决策循环 ─────────────────────────────────────────────────────────

    def _is_market_tradeable(self, features: dict) -> bool:
        """
        判断当前市场是否值得交易。

        两个条件均满足才放行：
        1. 全市场 volume z-score 绝对均值 ≥ min_market_activity（市场活跃）
        2. 全市场 5m 收益率绝对均值 ≥ min_market_move（存在价格结构）

        Why：横盘无趋势市场是横截面策略的最大亏损来源——
        所有币都在原地震荡，随机噪声会触发大量止损。
        """
        if not features:
            return False

        vols = [abs(f.volume_zscore) for f in features.values()]
        avg_vol = sum(vols) / len(vols)
        if avg_vol < self.params["min_market_activity"]:
            logger.info(f"[AlphaFactory] 市场活跃度不足 avg_vol_zscore={avg_vol:.2f}，跳过本轮")
            return False

        rets = [abs(f.ret_5m) for f in features.values() if f.ret_5m != 0]
        if rets:
            avg_move = sum(rets) / len(rets)
            if avg_move < self.params["min_market_move"]:
                logger.info(f"[AlphaFactory] 市场波动不足 avg_ret5m={avg_move:.4%}，跳过本轮")
                return False

        return True

    def _net_exposure(self) -> float:
        """
        当前多空名义价值净差额（正值=净多头，负值=净空头）。

        用于市场中性控制：防止单方向仓位过重而变成方向性赌注。
        """
        long_val  = sum(p["qty"] * p["entry_price"] for p in self.long_positions.values())
        short_val = sum(p["qty"] * p["entry_price"] for p in self.short_positions.values())
        return long_val - short_val

    def _calc_volatility(self, symbol: str) -> float:
        """
        计算单个品种的近期价格波动率（平均逐笔涨跌幅绝对值）。

        用于自适应 TP/SL：波动大的币给更宽的 TP/SL，避免被噪声止损。
        返回值已 clamp 在 [0.002, 0.025] 之间（0.2% ~ 2.5%）。
        """
        state = self.feature_engine.get_state(symbol)
        if state is None or len(state.price_series) < 3:
            return 0.005   # 数据不足时返回默认值

        n = self.params["vol_window"]
        prices = [p for _, p in list(state.price_series)[-n:]]
        if len(prices) < 2:
            return 0.005

        rets = [abs((prices[i] - prices[i-1]) / prices[i-1])
                for i in range(1, len(prices)) if prices[i-1] > 0]
        if not rets:
            return 0.005

        vol = sum(rets) / len(rets)
        return max(0.002, min(0.025, vol))

    def _evolve_factors(self):
        """
        基于近期交易的因子 IC 动态调整 ScoringEngine 权重（因子自动进化）。

        每 evolve_interval 轮排序调用一次。
        需要至少 evolve_min_trades 笔近期交易才触发，避免样本不足时的噪声更新。

        Why：因子的预测力会随市场状态变化，静态权重无法适应。
        通过 IC（因子分数与收益率的 Pearson 相关系数）实时评估，
        让表现好的因子权重增加，表现差的因子权重减少。
        """
        min_trades = self.params["evolve_min_trades"]
        if len(self._recent_trades) < min_trades:
            return

        trades = list(self._recent_trades)
        factor_trades = [t for t in trades if t.factors]
        if not factor_trades:
            return

        factor_names = [k for k in factor_trades[0].factors
                        if k != "total" and not k.endswith("_contrib")]
        rets  = [t.ret_lev_pct for t in factor_trades]
        n_f   = len(factor_trades)
        mean_r = sum(rets) / n_f

        def _dir(t, v):
            return v if t.side == "LONG" else -v

        ic_updates = {}
        for fname in factor_names:
            scores = [_dir(t, t.factors.get(fname, 0.0)) for t in factor_trades]
            mean_s = sum(scores) / n_f
            cov    = sum((s - mean_s) * (r - mean_r) for s, r in zip(scores, rets))
            std_s  = (sum((s - mean_s) ** 2 for s in scores) / n_f) ** 0.5
            std_r  = (sum((r - mean_r) ** 2 for r in rets)   / n_f) ** 0.5
            if std_s > 1e-8 and std_r > 1e-8:
                ic_updates[fname] = cov / n_f / (std_s * std_r)

        if ic_updates:
            self.scoring_engine.update_weights(ic_updates)
            self._export_ic_snapshot(ic_updates)

    # ─── 组合优化 ────────────────────────────────────────────────────────────

    def _get_portfolio_weight(self, symbol: str) -> float:
        """
        基于该品种历史 Sharpe proxy 返回仓位倍数，范围 [0.5, 1.5]。

        Sharpe proxy = mean(ret) / std(ret)（简化，无无风险利率）
        无历史数据时返回 1.0（中性，不影响原有仓位逻辑）。

        Why：过去赚钱的品种，alpha 来源更稳定，应给更多资本；
              过去亏钱的品种，应减小敞口或观察。
        """
        stats = self._symbol_stats.get(symbol)
        if not stats or len(stats) < self.params["port_min_trades"]:
            return 1.0

        rets = list(stats)
        n    = len(rets)
        mean = sum(rets) / n
        std  = (sum((r - mean) ** 2 for r in rets) / n) ** 0.5

        if std < 1e-8:
            return 1.0

        sharpe = mean / std
        # clamp sharpe 至 [-2, 2]，映射到倍数 [0.5, 1.5]
        scale  = self.params["port_weight_scale"]
        weight = 1.0 + scale * max(-1.0, min(1.0, sharpe / 2.0))
        return max(0.5, min(1.5, weight))

    def _compute_corr(self, sym1: str, sym2: str) -> float:
        """计算两个品种 ret_1m 历史的 Pearson 相关系数"""
        h1 = list(self._ret1m_history.get(sym1, []))
        h2 = list(self._ret1m_history.get(sym2, []))
        n  = min(len(h1), len(h2))
        if n < 10:
            return 0.0
        h1, h2 = h1[-n:], h2[-n:]
        m1, m2 = sum(h1) / n, sum(h2) / n
        cov = sum((a - m1) * (b - m2) for a, b in zip(h1, h2)) / n
        s1  = (sum((a - m1) ** 2 for a in h1) / n) ** 0.5
        s2  = (sum((b - m2) ** 2 for b in h2) / n) ** 0.5
        return (cov / (s1 * s2)) if s1 > 1e-8 and s2 > 1e-8 else 0.0

    def _dedup_by_correlation(self, candidates: list) -> list:
        """
        从候选列表中去除高度相关的品种（按 score 降序，保留排名更高的）。

        Why：BTC/ETH/SOL 常常同向波动，同时持有等于重复下注，
             去重后每个仓位携带更独立的 alpha 信息。
        """
        threshold = self.params["max_corr_threshold"]
        kept      = []
        for entry in candidates:
            correlated = any(
                abs(self._compute_corr(entry.symbol, k.symbol)) > threshold
                for k in kept
            )
            if not correlated:
                kept.append(entry)
            else:
                logger.debug(
                    f"[AlphaFactory] {entry.symbol} 与已选品种高度相关，跳过"
                )
        return kept

    # ─── 数据导出 ─────────────────────────────────────────────────────────────

    def _export_trade_factors(self, record: "TradeRecord"):
        """每笔平仓后追加因子记录到 factor_log.csv（供 Dashboard 使用）"""
        if not self.params["export_data"] or not record.factors:
            return
        try:
            rows = [
                [record.exit_time, record.symbol, record.side,
                 fname,
                 fval if record.side == "LONG" else -fval,
                 record.ret_lev_pct]
                for fname, fval in record.factors.items()
                if fname != "total" and not fname.endswith("_contrib")
            ]
            write_header = not os.path.exists(self._factor_log_path)
            with open(self._factor_log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["time", "symbol", "side", "factor", "value", "ret"])
                w.writerows(rows)
        except Exception as e:
            logger.debug(f"[AlphaFactory] factor_log 写入失败: {e}")

    def _export_ic_snapshot(self, ic_updates: Dict[str, float]):
        """每次因子进化后追加 IC 快照到 ic_log.csv（供 Dashboard 使用）"""
        if not self.params["export_data"]:
            return
        try:
            now     = time.time()
            weights = self.scoring_engine.get_current_weights()
            write_header = not os.path.exists(self._ic_log_path)
            with open(self._ic_log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["time", "factor", "ic", "weight"])
                for fname, ic in ic_updates.items():
                    w.writerow([now, fname, ic, weights.get(fname, 0.0)])
        except Exception as e:
            logger.debug(f"[AlphaFactory] ic_log 写入失败: {e}")

    def _export_positions_snapshot(self):
        """每轮排序后将当前持仓快照写入 positions_current.json（供 Dashboard 使用）"""
        if not self.params["export_data"]:
            return
        try:
            def _pos_row(symbol, pos, side):
                price = self._get_price(symbol)
                lev   = self.params["leverage"]
                if price > 0 and pos["entry_price"] > 0:
                    if side == "LONG":
                        ret_lev = (price - pos["entry_price"]) / pos["entry_price"] * lev * 100
                    else:
                        ret_lev = (pos["entry_price"] - price) / pos["entry_price"] * lev * 100
                else:
                    ret_lev = 0.0
                return {
                    "symbol":      symbol,
                    "entry_price": pos["entry_price"],
                    "score":       pos["score"],
                    "ret_lev_pct": round(ret_lev, 2),
                    "held_s":      round(time.time() - pos["entry_time"], 0),
                }

            data = {
                "ts":           time.time(),
                "long":         [_pos_row(s, p, "LONG")  for s, p in self.long_positions.items()],
                "short":        [_pos_row(s, p, "SHORT") for s, p in self.short_positions.items()],
                "net_exposure": round(self._net_exposure(), 2),
                "weights":      self.scoring_engine.get_current_weights(),
            }
            with open(self._positions_json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"[AlphaFactory] positions_current.json 写入失败: {e}")

    # ─── Regime filter ───────────────────────────────────────────────────────

    def _regime_filter(self, features: dict) -> dict:
        """
        基于 BTC 趋势判断市场制度，返回当前是否允许开多/开空。

        逻辑：
            BTC 5m 涨幅 > +threshold → 强势牛市 → 禁止新开空（顺势保护）
            BTC 5m 跌幅 > -threshold → 强势熊市 → 禁止新开多（顺势保护）
            否则 → 正常双向交易

        Why：
            横截面策略是 alpha，但在极端趋势行情下做反向会大量止损。
            此过滤器不阻止平仓，只阻止「新开仓」。
        """
        sym = self.params["regime_btc_symbol"]
        threshold = self.params["regime_5m_threshold"]
        feat = features.get(sym)

        if feat is None or feat.data_count < 20:
            return {"allow_long": True, "allow_short": True, "reason": "no_btc_data"}

        ret5m = feat.ret_5m

        if ret5m > threshold:
            return {
                "allow_long":  True,
                "allow_short": False,
                "reason":      f"btc_bull({ret5m:+.2%} 5m)",
            }
        if ret5m < -threshold:
            return {
                "allow_long":  False,
                "allow_short": True,
                "reason":      f"btc_bear({ret5m:+.2%} 5m)",
            }

        return {"allow_long": True, "allow_short": True, "reason": "normal"}

    def _rank_and_trade(self, now: float):
        """
        慢层：每 rank_interval 秒运行一次。

        职责：更新候选池（_candidate_pool），不直接开仓。
        实际开仓由快层 on_tick 中的 LOBTimingEngine 实时触发。
        """
        self._last_rank_time = now
        self._sync_positions()   # 检测交易所侧 TP/SL 是否已成交

        features = self.feature_engine.get_all_features()
        if len(features) < 10:
            return

        # 市场可交易性过滤（横盘磨损区直接跳过）
        if not self._is_market_tradeable(features):
            return

        scores = self.scoring_engine.compute_scores(features)
        if not scores:
            return

        # Dispersion Filter：截面分散度不足 → 无 alpha，跳过候选池更新
        score_vals = list(scores.values())
        n_s = len(score_vals)
        if n_s > 1:
            mean_s = sum(score_vals) / n_s
            disp   = (sum((v - mean_s) ** 2 for v in score_vals) / n_s) ** 0.5
            if disp < self.params["min_score_dispersion"]:
                logger.info(f"[AlphaFactory] 截面分散度不足 std={disp:.3f}，跳过候选池更新")
                self._log_positions()
                self._export_positions_snapshot()
                return

        # 快照 ret_1m 历史（用于相关性去重）
        _corr_len = self.params["corr_history_len"]
        for sym, feat in features.items():
            if sym not in self._ret1m_history:
                self._ret1m_history[sym] = deque(maxlen=_corr_len)
            self._ret1m_history[sym].append(feat.ret_1m)

        # 因子权重进化（每 evolve_interval 轮触发一次）
        if self.ranking_engine.rank_count % self.params["evolve_interval"] == 0:
            self._evolve_factors()

        (long_top, _, _, short_top, _, _) = self.ranking_engine.rank(scores)

        # 热身期：让 EMA 和候选池先稳定
        if self.ranking_engine.rank_count <= self.params["warmup_count"]:
            logger.info(
                f"[AlphaFactory] 热身期 "
                f"({self.ranking_engine.rank_count}/{self.params['warmup_count']})"
            )
            return

        # Kill Switch：冲击过多时清空候选池（禁止 on_tick 开新仓）
        if self.shock_detector.is_kill_switched():
            status = self.shock_detector.get_status()
            logger.warning(
                f"[ShockDetector] Kill Switch 触发，清空候选池 "
                f"(resume_in={status['kill_resume_in']:.0f}s)"
            )
            self._candidate_pool = {"long": set(), "short": set()}
            return

        # 相关性去重
        long_top  = self._dedup_by_correlation(long_top)
        short_top = self._dedup_by_correlation(short_top)

        # 更新候选池（原子赋值，GIL 保护线程安全）
        self._candidate_pool = {
            "long":  {e.symbol for e in long_top},
            "short": {e.symbol for e in short_top},
        }
        # 更新特征和得分快照（供 on_tick 使用）
        self._latest_features = features
        self._latest_scores   = scores

        logger.debug(
            f"[AlphaFactory] 候选池更新 "
            f"多头={list(self._candidate_pool['long'])[:5]} "
            f"空头={list(self._candidate_pool['short'])[:5]}"
        )

        self._log_positions()
        self._export_positions_snapshot()

    # ─── 开仓 / 平仓 ─────────────────────────────────────────────────────────

    def _open_long(self, symbol: str, score: float, price: float, features: dict = None):
        if price <= 0:
            return

        # 市场中性：净多头敞口超限时拒绝新开多
        net_exp = self._net_exposure()
        if net_exp > self.params["max_net_exposure"]:
            logger.info(f"[AlphaFactory] {symbol} LONG 跳过：净多头敞口 {net_exp:.0f} 超限")
            return

        # 动态仓位：信号越强，仓位越大（score 经横截面归一化，典型范围 ±3σ）
        score_abs   = min(abs(score), 3.0)
        min_s, max_s = self.params["size_min_scale"], self.params["size_max_scale"]
        size_scale  = min_s + (max_s - min_s) * (score_abs / 3.0)
        # 组合优化：历史 Sharpe proxy 进一步调整仓位
        port_w     = self._get_portfolio_weight(symbol)
        trade_usdt = self.params["trade_size_usdt"] * size_scale * port_w
        qty = round(trade_usdt / price, 6)
        if qty <= 0 or qty * price < 5.0:
            logger.debug(f"[AlphaFactory] {symbol} 名义价值不足，跳过开多")
            return

        # 波动率自适应 TP/SL
        vol      = self._calc_volatility(symbol)
        sl_price_vol = price * (1 - vol * self.params["sl_vol_mult"])

        result = self.market_buy(symbol, qty, reduce_only=False)
        if result is not None:
            factors = self.scoring_engine.get_factor_breakdown(symbol, features) if features else {}
            self.long_positions[symbol] = {
                "entry_price": price,
                "qty":         qty,
                "entry_time":  time.time(),
                "score":       score,
                "factors":     factors or {},
                "max_price":   price,   # trailing stop 用：记录持仓期间最高价
                "max_score":   score,   # alpha peak exit 用：记录持仓期间最高得分
            }
            sl_price = sl_price_vol
            sl_res = self.stop_order(symbol, "SELL", "STOP_MARKET", sl_price)
            self._tp_sl_orders[symbol] = {
                "sl_id":    sl_res.get("orderId") if sl_res else None,
                "tp_id":    None,
                "sl_price": sl_price,
                "tp_price": None,
            }
            logger.info(
                f"[AlphaFactory] ▲ LONG  {symbol} "
                f"qty={qty:.4f}({size_scale:.0%}) price={price:.4f} score={score:+.3f} "
                f"vol={vol:.4%} SL={sl_price:.4f}({vol*self.params['sl_vol_mult']:.2%}) "
                f"[signal-exit mode]"
            )

    def _open_short(self, symbol: str, score: float, price: float, features: dict = None):
        if price <= 0:
            return

        # 市场中性：净空头敞口超限时拒绝新开空
        net_exp = self._net_exposure()
        if net_exp < -self.params["max_net_exposure"]:
            logger.info(f"[AlphaFactory] {symbol} SHORT 跳过：净空头敞口 {net_exp:.0f} 超限")
            return

        # 动态仓位：信号越强，仓位越大
        score_abs   = min(abs(score), 3.0)
        min_s, max_s = self.params["size_min_scale"], self.params["size_max_scale"]
        size_scale  = min_s + (max_s - min_s) * (score_abs / 3.0)
        # 组合优化：历史 Sharpe proxy 进一步调整仓位
        port_w     = self._get_portfolio_weight(symbol)
        trade_usdt = self.params["trade_size_usdt"] * size_scale * port_w
        qty = round(trade_usdt / price, 6)
        if qty <= 0 or qty * price < 5.0:
            logger.debug(f"[AlphaFactory] {symbol} 名义价值不足，跳过开空")
            return

        # 硬止损（防爆仓底线）
        vol      = self._calc_volatility(symbol)
        sl_price_vol = price * (1 + vol * self.params["sl_vol_mult"])

        result = self.market_sell(symbol, qty, reduce_only=False)
        if result is not None:
            factors = self.scoring_engine.get_factor_breakdown(symbol, features) if features else {}
            self.short_positions[symbol] = {
                "entry_price": price,
                "qty":         qty,
                "entry_time":  time.time(),
                "score":       score,
                "factors":     factors or {},
                "min_price":   price,   # trailing stop 用：记录持仓期间最低价
                "max_score":   abs(score),  # alpha peak exit 用：记录持仓期间最大得分绝对值
            }
            sl_price = sl_price_vol
            sl_res = self.stop_order(symbol, "BUY", "STOP_MARKET", sl_price)
            self._tp_sl_orders[symbol] = {
                "sl_id":    sl_res.get("orderId") if sl_res else None,
                "tp_id":    None,
                "sl_price": sl_price,
                "tp_price": None,
            }
            logger.info(
                f"[AlphaFactory] ▼ SHORT {symbol} "
                f"qty={qty:.4f}({size_scale:.0%}) price={price:.4f} score={score:+.3f} "
                f"vol={vol:.4%} SL={sl_price:.4f}({vol*self.params['sl_vol_mult']:.2%}) "
                f"[signal-exit mode]"
            )

    def _close_long(self, symbol: str, reason: str):
        pos = self.long_positions.pop(symbol, None)
        if pos is None:
            return
        self._cancel_tp_sl(symbol)
        self._qgate_fails.pop(symbol, None)
        now        = time.time()
        exit_price = self._get_price(symbol)
        lev        = self.params["leverage"]
        fee_rate   = self.params["fee_rate"]
        ret        = ((exit_price - pos["entry_price"]) / pos["entry_price"]) if exit_price > 0 else 0
        ret_lev    = ret * lev
        pnl_usdt   = pos["qty"] * pos["entry_price"] * ret
        fee_usdt   = pos["qty"] * (pos["entry_price"] + (exit_price if exit_price > 0 else pos["entry_price"])) * fee_rate
        held       = now - pos["entry_time"]
        self.market_sell(symbol, pos["qty"], reduce_only=True)
        record = TradeRecord(
            symbol=symbol, side="LONG",
            entry_price=pos["entry_price"], exit_price=exit_price,
            qty=pos["qty"], entry_time=pos["entry_time"], exit_time=now,
            reason=reason.split("(")[0], leverage=lev,
            pnl_usdt=pnl_usdt, ret_pct=ret * 100,
            ret_lev_pct=ret_lev * 100, hold_seconds=held,
            fee_usdt=fee_usdt,
            factors=pos.get("factors", {}),
        )
        self._trades.append(record)
        self._recent_trades.append(record)
        # 组合优化：更新该品种历史收益率
        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = deque(maxlen=self.params["port_stats_window"])
        self._symbol_stats[symbol].append(ret_lev * 100)
        self._export_trade_factors(record)
        if self._db:
            self._db.save_completed_trade(record)
        self.ranking_engine.release_long(symbol)
        # 记录多头平仓时间，供反向开仓冷却期使用
        self._recently_closed.setdefault(symbol, {})["LONG"] = now
        net_pnl = pnl_usdt - fee_usdt
        logger.info(
            f"[AlphaFactory] ■ CLOSE LONG  {symbol} reason={reason} "
            f"pnl={ret*100:+.2f}% (保证金{ret_lev*100:+.2f}%) "
            f"毛={pnl_usdt:+.3f} 费={fee_usdt:.3f} 净={net_pnl:+.3f}USDT held={held:.0f}s"
        )

    def _close_short(self, symbol: str, reason: str):
        pos = self.short_positions.pop(symbol, None)
        if pos is None:
            return
        self._cancel_tp_sl(symbol)
        self._qgate_fails.pop(symbol, None)
        now        = time.time()
        exit_price = self._get_price(symbol)
        lev        = self.params["leverage"]
        fee_rate   = self.params["fee_rate"]
        ret        = ((pos["entry_price"] - exit_price) / pos["entry_price"]) if exit_price > 0 else 0
        ret_lev    = ret * lev
        pnl_usdt   = pos["qty"] * pos["entry_price"] * ret
        fee_usdt   = pos["qty"] * (pos["entry_price"] + (exit_price if exit_price > 0 else pos["entry_price"])) * fee_rate
        held       = now - pos["entry_time"]
        self.market_buy(symbol, pos["qty"], reduce_only=True)
        record = TradeRecord(
            symbol=symbol, side="SHORT",
            entry_price=pos["entry_price"], exit_price=exit_price,
            qty=pos["qty"], entry_time=pos["entry_time"], exit_time=now,
            reason=reason.split("(")[0], leverage=lev,
            pnl_usdt=pnl_usdt, ret_pct=ret * 100,
            ret_lev_pct=ret_lev * 100, hold_seconds=held,
            fee_usdt=fee_usdt,
            factors=pos.get("factors", {}),
        )
        self._trades.append(record)
        self._recent_trades.append(record)
        # 组合优化：更新该品种历史收益率
        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = deque(maxlen=self.params["port_stats_window"])
        self._symbol_stats[symbol].append(ret_lev * 100)
        self._export_trade_factors(record)
        if self._db:
            self._db.save_completed_trade(record)
        self.ranking_engine.release_short(symbol)
        # 记录空头平仓时间，供反向开仓冷却期使用
        self._recently_closed.setdefault(symbol, {})["SHORT"] = now
        net_pnl = pnl_usdt - fee_usdt
        logger.info(
            f"[AlphaFactory] ■ CLOSE SHORT {symbol} reason={reason} "
            f"pnl={ret*100:+.2f}% (保证金{ret_lev*100:+.2f}%) "
            f"毛={pnl_usdt:+.3f} 费={fee_usdt:.3f} 净={net_pnl:+.3f}USDT held={held:.0f}s"
        )

    # ─── TP/SL 单管理 ────────────────────────────────────────────────────────

    def _cancel_tp_sl(self, symbol: str):
        """撤销该品种挂着的 TP/SL 条件单（在策略主动平仓前调用）"""
        orders = self._tp_sl_orders.pop(symbol, None)
        if not orders:
            return
        for key in ("sl_id", "tp_id"):
            oid = orders.get(key)
            if oid:
                try:
                    self.cancel_order(symbol, oid)
                except Exception:
                    pass  # 单子可能已成交或不存在，忽略撤单失败

    def _sync_positions(self):
        """
        与交易所持仓同步：检测被 TP/SL 条件单成交而关闭的仓位。
        每次排序时调用（约每 rank_interval 秒一次）。
        仅在引擎提供 get_positions() 时生效（PaperEngine 跳过）。
        """
        if not hasattr(self.engine, "get_positions"):
            return
        try:
            live = {p["symbol"] for p in self.engine.get_positions()}
        except Exception:
            return

        for sym in list(self.long_positions.keys()):
            if sym not in live:
                self._record_exchange_close(sym, "LONG")

        for sym in list(self.short_positions.keys()):
            if sym not in live:
                self._record_exchange_close(sym, "SHORT")

    def _record_exchange_close(self, symbol: str, side: str):
        """
        交易所侧 TP/SL 成交后，补记 TradeRecord 并清理策略状态。
        同时撤销另一边未成交的条件单（OCO 行为）。
        """
        positions = self.long_positions if side == "LONG" else self.short_positions
        pos = positions.pop(symbol, None)
        if pos is None:
            return

        # 先 peek 拿到价格信息（用于推断触发原因），再通过 _cancel_tp_sl 撤掉对面单
        orders = self._tp_sl_orders.get(symbol)   # peek，不 pop
        self._cancel_tp_sl(symbol)                 # 撤销另一边挂单并 pop
        now           = time.time()
        current_price = self._get_price(symbol)
        lev           = self.params["leverage"]

        # 用预设的条件单价格作为退出价（而非当前市价）
        # 当前市价可能在 _sync_positions 轮询延迟内已大幅偏移，导致收益率失真
        if orders:
            tp_price = orders.get("tp_price")
            sl_price = orders.get("sl_price")
            # tp_price 可能为 None（信号驱动模式不挂 TP）
            if tp_price is not None and sl_price is not None:
                tp_dist = abs(current_price - tp_price)
                sl_dist = abs(current_price - sl_price)
                if tp_dist <= sl_dist:
                    reason, exit_price = "take_profit", tp_price
                else:
                    reason, exit_price = "stop_loss",   sl_price
            elif sl_price is not None:
                reason, exit_price = "stop_loss", sl_price
            elif tp_price is not None:
                reason, exit_price = "take_profit", tp_price
            else:
                exit_price = current_price
                reason     = "strategy_stop"
        else:
            exit_price = current_price
            reason     = None   # 后面根据 ret_lev 再定

        if exit_price > 0 and pos["entry_price"] > 0:
            if side == "LONG":
                ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
            else:
                ret = (pos["entry_price"] - exit_price) / pos["entry_price"]
        else:
            ret = 0.0

        fee_rate = self.params["fee_rate"]
        ret_lev  = ret * lev
        pnl_usdt = pos["qty"] * pos["entry_price"] * ret
        fee_usdt = pos["qty"] * (pos["entry_price"] + (exit_price if exit_price > 0 else pos["entry_price"])) * fee_rate
        held     = now - pos["entry_time"]

        if reason is None:
            reason = "take_profit" if ret_lev > 0 else "stop_loss"

        record = TradeRecord(
            symbol=symbol, side=side,
            entry_price=pos["entry_price"], exit_price=exit_price,
            qty=pos["qty"], entry_time=pos["entry_time"], exit_time=now,
            reason=reason, leverage=lev,
            pnl_usdt=pnl_usdt, ret_pct=ret * 100,
            ret_lev_pct=ret_lev * 100, hold_seconds=held,
            fee_usdt=fee_usdt,
            factors=pos.get("factors", {}),
        )
        self._trades.append(record)
        self._recent_trades.append(record)
        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = deque(maxlen=self.params["port_stats_window"])
        self._symbol_stats[symbol].append(ret_lev * 100)
        self._export_trade_factors(record)
        if self._db:
            self._db.save_completed_trade(record)

        if side == "LONG":
            self.ranking_engine.release_long(symbol)
        else:
            self.ranking_engine.release_short(symbol)
        self._qgate_fails.pop(symbol, None)

        net_pnl = pnl_usdt - fee_usdt
        logger.info(
            f"[AlphaFactory] ■ {side} {symbol} 交易所侧平仓({reason}) "
            f"pnl={ret*100:+.2f}% (保证金{ret_lev*100:+.2f}%) "
            f"毛={pnl_usdt:+.3f} 费={fee_usdt:.3f} 净={net_pnl:+.3f}USDT held={held:.0f}s"
        )

    # ─── 辅助 ────────────────────────────────────────────────────────────────

    def _pass_quality_gate(self, symbol: str, feat, side: str) -> bool:
        """
        进场质量门槛（多空共用）。

        三层过滤：
          0. 冲击暂停（ShockDetector 标记期间禁止开仓）
          1. 流动性基础（成交量 & 价差）
          2. 反向开仓冷却期（防止频繁 flip）

        进场方向由 LOBTimingEngine 负责（timing_score 超阈值），
        此处只做流动性和冷却期的基础过滤。
        """
        if feat is None:
            return False

        # ── 0. 冲击暂停检查（被 ShockDetector 标记的品种禁止开仓）────────────
        if self.shock_detector.is_paused(symbol):
            logger.debug(f"[AlphaFactory] {symbol} {side} 冲击暂停窗口，跳过")
            return False

        # ── 1. 流动性基础检查 ────────────────────────────────────────────────
        if abs(feat.volume_zscore) < self.params["min_volume_zscore"]:
            logger.info(
                f"[AlphaFactory] {symbol} {side} 量不足 "
                f"zscore={feat.volume_zscore:.2f} (需>{self.params['min_volume_zscore']})"
            )
            return False
        if feat.spread_bps > self.params["max_spread_bps"]:
            logger.info(
                f"[AlphaFactory] {symbol} {side} 价差过大 "
                f"{feat.spread_bps:.1f}bps (需<{self.params['max_spread_bps']}bps)"
            )
            return False

        # ── 2. 反向开仓冷却期：防止频繁 flip ─────────────────────────────────
        cooldown = self.params["cooldown_seconds"]
        if cooldown > 0:
            opposite   = "SHORT" if side == "LONG" else "LONG"
            last_close = self._recently_closed.get(symbol, {}).get(opposite, 0.0)
            elapsed    = time.time() - last_close
            if elapsed < cooldown:
                logger.info(
                    f"[AlphaFactory] {symbol} {side} 反向冷却期 "
                    f"({elapsed:.0f}s/{cooldown}s)，跳过"
                )
                return False

        return True

    def _get_price(self, symbol: str) -> float:
        state = self.feature_engine.get_state(symbol)
        if state and state.price_series:
            return state.price_series[-1][1]
        return 0.0

    def _log_positions(self):
        if not self.long_positions and not self.short_positions:
            return
        lev = self.params["leverage"]

        def _ret_str(symbol: str, pos: dict, side: str) -> str:
            price = self._get_price(symbol)
            if price > 0 and pos["entry_price"] > 0:
                if side == "LONG":
                    ret_lev = (price - pos["entry_price"]) / pos["entry_price"] * lev * 100
                else:
                    ret_lev = (pos["entry_price"] - price) / pos["entry_price"] * lev * 100
                return f"{ret_lev:+.1f}%"
            return "n/a"

        longs  = " ".join(
            f"▲{s}({_ret_str(s, pos, 'LONG')})"
            for s, pos in self.long_positions.items()
        )
        shorts = " ".join(
            f"▼{s}({_ret_str(s, pos, 'SHORT')})"
            for s, pos in self.short_positions.items()
        )
        logger.info(f"[AlphaFactory] 持仓: {longs}  {shorts}".strip())

    # ─── 生命周期 ────────────────────────────────────────────────────────────

    def on_start(self):
        logger.info(f"[AlphaFactory] 策略启动 | 监控 {len(self.symbols)} 个品种")

    def on_stop(self):
        logger.info("[AlphaFactory] 策略停止，全部清仓...")
        for sym in list(self.long_positions.keys()):
            self._close_long(sym, "strategy_stop")
        for sym in list(self.short_positions.keys()):
            self._close_short(sym, "strategy_stop")
        self._print_report()

    def _print_report(self):
        trades = self._trades
        sep = "=" * 60

        if not trades:
            logger.info(f"\n{sep}\n  交易报表：本次运行无已平仓交易\n{sep}")
            return

        n          = len(trades)
        total_fee  = sum(t.fee_usdt for t in trades)
        # 用净 PnL 判断盈亏（手续费后才是真实盈亏）
        wins       = [t for t in trades if t.pnl_usdt - t.fee_usdt > 0]
        losses     = [t for t in trades if t.pnl_usdt - t.fee_usdt <= 0]
        win_rate   = len(wins) / n * 100
        total_pnl  = sum(t.pnl_usdt for t in trades)
        total_net  = total_pnl - total_fee
        avg_ret    = sum(t.ret_lev_pct for t in trades) / n
        avg_hold   = sum(t.hold_seconds for t in trades) / n

        best  = max(trades, key=lambda t: t.ret_lev_pct)
        worst = min(trades, key=lambda t: t.ret_lev_pct)

        longs  = [t for t in trades if t.side == "LONG"]
        shorts = [t for t in trades if t.side == "SHORT"]
        tps    = [t for t in trades if t.reason == "take_profit"]
        sls    = [t for t in trades if t.reason == "stop_loss"]
        stops  = [t for t in trades if t.reason == "strategy_stop"]

        avg_win  = sum(t.ret_lev_pct for t in wins)  / len(wins)  if wins   else 0
        avg_loss = sum(t.ret_lev_pct for t in losses) / len(losses) if losses else 0
        net_wins   = sum(t.pnl_usdt - t.fee_usdt for t in wins)
        net_losses = sum(t.pnl_usdt - t.fee_usdt for t in losses)
        profit_factor = (
            abs(net_wins / net_losses)
            if losses and net_losses != 0 else float("inf")
        )

        lines = [
            sep,
            "  Alpha Factory 交易报表",
            sep,
            f"  运行轮次     : {self.ranking_engine.rank_count} 次排序 | Tick {self._tick_count:,}",
            f"  总平仓笔数   : {n}  (多 {len(longs)} 空 {len(shorts)})",
            f"  止盈触发     : {len(tps)} 笔  | 止损触发 : {len(sls)} 笔  | 强平收仓 : {len(stops)} 笔",
            sep,
            f"  胜率         : {win_rate:.1f}%  ({len(wins)}胜 / {len(losses)}败)",
            f"  盈亏比       : {profit_factor:.2f}  (平均盈 {avg_win:+.1f}% / 平均亏 {avg_loss:+.1f}%)",
            f"  累计毛 PnL   : {total_pnl:+.4f} USDT",
            f"  累计手续费   : -{total_fee:.4f} USDT  (taker {self.params['fee_rate']*100:.3f}%×2×{n}笔)",
            f"  累计净 PnL   : {total_net:+.4f} USDT  ← 真实盈亏",
            f"  平均保证金回报: {avg_ret:+.2f}% / 笔",
            f"  平均持仓时间 : {avg_hold:.0f} 秒  ({avg_hold/60:.1f} 分钟)",
            sep,
            f"  最佳交易 : {best.symbol} {best.side}  保证金 {best.ret_lev_pct:+.2f}%  {best.reason}",
            f"  最差交易 : {worst.symbol} {worst.side}  保证金 {worst.ret_lev_pct:+.2f}%  {worst.reason}",
            sep,
            "  ── 逐笔明细 ──────────────────────────────────────────",
            f"  {'品种':<14} {'方向':<6} {'保证金%':>8} {'USDT':>9} {'持仓(s)':>8} {'原因'}",
        ]
        for t in sorted(trades, key=lambda x: x.entry_time):
            lines.append(
                f"  {t.symbol:<14} {t.side:<6} "
                f"{t.ret_lev_pct:>+7.2f}% "
                f"{t.pnl_usdt:>+9.4f} "
                f"{t.hold_seconds:>7.0f}s  "
                f"{t.reason}"
            )
        lines.append(sep)

        # ── 因子盈利能力分析 ──────────────────────────────────────────────────
        factor_trades = [t for t in trades if t.factors]
        if factor_trades:
            # 只取原始 Z-score 字段（排除 _contrib / total）
            factor_names = [k for k in factor_trades[0].factors
                            if k != "total" and not k.endswith("_contrib")]
            live_weights = self.scoring_engine.get_current_weights()
            lines += [
                "  ── 因子盈利能力分析（方向调整后Z-score）────────────────────",
                "  * 做空时因子取反，使「正值=有利信号」对多空统一",
                f"  {'因子':<20} {'权重(当前)':>10} {'IC(预测力)':>12} {'赢家Z均值':>10} {'输家Z均值':>10} {'差值':>8}",
            ]
            # 方向调整：做空时因子取反，确保「正值=该方向有利信号」
            def _dir(t: "TradeRecord", v: float) -> float:
                return v if t.side == "LONG" else -v

            rets = [t.ret_lev_pct for t in factor_trades]
            for fname in factor_names:
                scores = [_dir(t, t.factors.get(fname, 0.0)) for t in factor_trades]
                # IC：方向调整后因子得分与收益率的 Pearson 相关系数
                n_f = len(scores)
                if n_f > 1:
                    mean_s = sum(scores) / n_f
                    mean_r = sum(rets) / n_f
                    cov    = sum((s - mean_s) * (r - mean_r) for s, r in zip(scores, rets))
                    std_s  = (sum((s - mean_s) ** 2 for s in scores) / n_f) ** 0.5
                    std_r  = (sum((r - mean_r) ** 2 for r in rets)   / n_f) ** 0.5
                    ic     = (cov / n_f / (std_s * std_r)) if std_s > 1e-8 and std_r > 1e-8 else 0.0
                else:
                    ic = 0.0
                win_scores  = [_dir(t, t.factors.get(fname, 0.0)) for t in factor_trades if t.pnl_usdt > 0]
                loss_scores = [_dir(t, t.factors.get(fname, 0.0)) for t in factor_trades if t.pnl_usdt <= 0]
                avg_win_f   = sum(win_scores)  / len(win_scores)  if win_scores  else 0.0
                avg_loss_f  = sum(loss_scores) / len(loss_scores) if loss_scores else 0.0
                weight = live_weights.get(fname, 0.0)
                lines.append(
                    f"  {fname:<20} {weight:>9.3f}  {ic:>+11.3f}  {avg_win_f:>+9.3f}  {avg_loss_f:>+9.3f}  {avg_win_f-avg_loss_f:>+7.3f}"
                )
            lines.append(sep)

        for line in lines:
            logger.info(line)

    def get_status(self) -> dict:
        return {
            "long_positions":  dict(self.long_positions),
            "short_positions": dict(self.short_positions),
            "long_top":        self.ranking_engine.current_longs,
            "short_top":       self.ranking_engine.current_shorts,
            "rank_count":      self.ranking_engine.rank_count,
            "tick_count":      self._tick_count,
            "active_symbols":  len(self.feature_engine.active_symbols),
        }

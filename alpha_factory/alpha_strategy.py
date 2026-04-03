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
import math
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
from .alpha_lifecycle import AlphaLifecycleTracker, AlphaState
from .trade_recorder import (
    TradeRecorder, TradeEvent,
    EVENT_RANK_SNAPSHOT, EVENT_TRAILING_ARMED, EVENT_TRAILING_HIT, EVENT_PEAK_PNL_UPDATE,
)
from .market_state_engine import MarketStateEngine
from .alpha_fusion import AlphaFusionEngine
from portfolio.portfolio_constructor import PortfolioConstructor
from execution.cost_model import CostModel


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
        "long_score_threshold":  0.8,  # 候选池做多最低 EMA 分数（提高门槛，降低交易频率）
        "short_score_threshold":-0.8,  # 候选池做空最高 EMA 分数（提高门槛，降低交易频率）
        "ema_alpha":             0.4,  # EMA 平滑系数
        "confirm_rounds":        3,    # 连续上榜 N 次才进入候选池（原 2，提高信号确认要求）
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
        "timing_entry_threshold": 0.40,  # timing_score 超此值 → 触发开仓（microprice 单独强即可）
        "timing_exit_threshold":  0.340,  # timing_score 反向超此值 → alpha_flip 平仓（死区 [-0.15, +0.30]）
        "alpha_flip_confirm_secs": 3.0,  # alpha_flip 需持续反向 N 秒才真正平仓（防盘口抖动）
        # ── Regime filter ────────────────────────────────────────────────────
        "regime_btc_symbol":  "BTCUSDT",  # 用于判断市场制度的参考品种
        "regime_5m_threshold": 0.02,  # BTC 5分钟涨跌超此值（2%）→ 触发方向性过滤
        # ── 市场可交易性过滤 ─────────────────────────────────────────────────
        "min_market_activity": 0.3,   # 全市场 vol_zscore 绝对均值最小值（低于此=市场静止）
        "min_market_move":     0.001, # 全市场 5m 平均涨跌幅最小值（低于此=横盘磨损区）
        # ── 市场中性敞口控制 ─────────────────────────────────────────────────
        "max_net_exposure":  300.0,   # 多空名义价值最大净差额（USDT），超出则不再加方向
        # ── Alpha Trailing（动态止盈：利润锁）───────────────────────────────────
        "alpha_trailing_min_pnl":          0.003,  # 激活阈值：价格收益率 ≥ 0.3%
        "alpha_trailing_expansion_ratio":  0.55,   # EXPANSION 阶段允许峰值回撤 55%（偏保守，手续费敏感阶段）
        "alpha_trailing_decay_ratio":      0.35,   # DECAY 阶段收紧至峰值回撤 35%
        # ── Layer 1：alpha 置信度驱动的动态止损 ─────────────────────────────
        "sl_vol_mult":         3.5,   # 兼容旧接口，保留但仅作 fallback（已被 dynamic stop 替代）
        "vol_window":           20,   # 计算波动率用的价格序列长度（笔数）
        # dynamic stop: sl_dist = open_vol × (base_mult + conf_mult × confidence)
        # confidence ∈ [0, 1]，越高 → stop 越宽（强 alpha 多给空间）
        "sl_base_mult":        2.0,   # confidence=0 时最紧止损倍数（open_vol × 2.0）
        "sl_conf_mult":        3.0,   # confidence=1 时额外宽度倍数（最宽 = (2.0+3.0)×vol）
        # low-confidence early cut：confidence 持续偏低 + 已有不利浮亏 → 提前退出
        "low_conf_threshold":  0.25,  # confidence 低于此值视为"低置信"
        "low_conf_adverse_loss": 0.005, # 不利浮亏超此值（价格收益率）才触发提前止损
        "low_conf_rounds":      4,    # 连续 N 轮低置信后触发（防止单点噪声误出场）
        # early-reversal in DECAY：DECAY 阶段 confidence 持续偏低 + 负动量 → 不等 REVERSAL 直接退出
        "early_reversal_conf_thresh": 0.30,  # DECAY 阶段 confidence 低于此值触发计数
        "early_reversal_rounds":       3,    # 连续 N 轮满足条件 → early_reversal 退出
        # ── Layer 3：Lifecycle 状态感知 Trailing（DECAY 收紧，EXPANSION 宽松）─
        "trailing_min_profit": 0.004, # 激活 trailing 所需最低盈利（价格收益率）
        "trailing_expansion_ratio": 0.50, # EXPANSION：允许峰值价格回撤 50%×vol
        "trailing_decay_ratio":     0.25, # DECAY：收紧至峰值价格回撤 25%×vol
        # ── Layer 4：时间退出（无收益仓位清理）──────────────────────────────
        "max_hold_seconds":    3600,  # 最长持仓时间（秒），超过且无盈利则退出
        "time_exit_min_profit": 0.001, # 时间退出豁免阈值：盈利 > 此值则继续持有
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
        # ── 终极版新参数 ──────────────────────────────────────────────────────────
        "min_edge_multiple":   1.2,   # 预期 alpha 收益必须是成本的 N 倍
        "lifecycle_exit":      True,  # 是否启用 alpha 生命周期驱动退出（REVERSAL 状态）
        "kill_switch_enabled": True,  # Kill Switch 开关（False=完全关闭，调试/测试用）
        "shock_detector_enabled": True,  # ShockDetector 开关（False=关闭所有冲击检测和 shock_exit）
        # ── 防锁死：连续无交易时自动放宽阈值 ─────────────────────────────────────
        "no_trade_relax_rounds":   3,    # 连续 N 轮无交易后进入放宽模式
        "relax_timing_factor":     0.65, # 放宽模式下 timing 入场阈值乘以此系数（降低要求）
        "relax_edge_factor":       0.75, # 放宽模式下 min_edge_multiple 乘以此系数
        "relax_min_vol_factor":    0.70, # 放宽模式下 min_volume_zscore 乘以此系数
        # ── 调试模式：买入后不卖出 ────────────────────────────────────────────
        "hold_forever":            False,  # True=禁止所有平仓（调试/观察用）
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

        # ── 终极版新模块 ──────────────────────────────────────────────────────
        self.market_state_engine = MarketStateEngine(
            btc_symbol = merged.get("regime_btc_symbol", "BTCUSDT")
        )
        self.alpha_fusion = AlphaFusionEngine(
            entry_threshold = merged.get("long_score_threshold", 0.30)
        )
        self.lifecycle_tracker = AlphaLifecycleTracker()
        self.portfolio_constructor = PortfolioConstructor(
            max_long_positions  = merged.get("max_long_positions", 3),
            max_short_positions = merged.get("max_short_positions", 3),
            max_net_exposure    = merged.get("max_net_exposure", 300.0),
            base_size_usdt      = merged.get("trade_size_usdt", 100.0),
            min_alpha_scale     = merged.get("size_min_scale", 0.7),
            max_alpha_scale     = merged.get("size_max_scale", 1.5),
            max_corr_threshold  = merged.get("max_corr_threshold", 0.85),
            corr_history_len    = merged.get("corr_history_len", 30),
            port_weight_scale   = merged.get("port_weight_scale", 0.4),
            port_min_trades     = merged.get("port_min_trades", 5),
        )
        self.cost_model = CostModel(
            fee_rate          = merged.get("fee_rate", 0.0004),
            min_edge_multiple = merged.get("min_edge_multiple", 1.5),
        )

        # 最新市场状态（每次慢层排名后更新，供 on_tick 查询）
        self._latest_market_state = None
        # 最新融合 alpha（{symbol: FusedAlpha}，供 Dashboard 和归因使用）
        self._latest_fused_alphas: dict = {}

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

        # ── depth5 动态订阅钩子（由外部 Runner 注入 ws_feed.update_depth_symbols）
        self._depth_update_fn = None

        # ── 运行时黑名单（外部 Runner 发现无效合约时写入，策略层自动过滤）
        self._symbol_blacklist: set = set()

        # ── 防锁死：追踪无交易轮数
        self._last_open_time:   float = 0.0   # 最近一次开仓时间
        self._no_trade_rounds:  int   = 0     # 连续无交易的排序轮数

        # ── 外部排序线程模式（testnet runner 使用独立 rank_loop 线程时设为 True）
        # 为 True 时 on_tick 跳过内部 rank 触发，由外部线程负责定时调用 _rank_and_trade
        self._external_rank_loop: bool = False

        # 组合优化：每个品种的历史收益率（用于 Sharpe proxy）
        self._symbol_stats:    Dict[str, deque] = {}
        # 相关性去重：每个品种的 ret_1m 快照历史
        self._ret1m_history:   Dict[str, deque] = {}

        # 数据导出：初始化文件路径
        _dir = self.params["export_dir"]
        self._ic_log_path        = os.path.join(_dir, "ic_log.csv")
        self._positions_json_path = os.path.join(_dir, "positions_current.json")
        self._trail_dir          = os.path.join(_dir, "price_trails")
        os.makedirs(self._trail_dir, exist_ok=True)

        # TradeRecorder：持仓事件轨迹 + 三张全局归因表
        self.trade_recorder = TradeRecorder(
            export_dir = _dir,
            reset      = self.params.get("recorder_reset", False),
        )

        # 价格轨迹录制：{symbol: {"side", "entry_price", "sl_price", "tp_price", "points": [(ts, price)], "last_record_ts"}}
        self._price_trails: Dict[str, dict] = {}
        # 追踪每个确认品种连续质量门槛失败次数；超过阈值后从确认池驱逐，腾出位置
        self._qgate_fails: Dict[str, int] = {}
        # 平仓时间记录：{symbol: {"LONG": ts, "SHORT": ts}}，用于反向开仓冷却期计算
        self._recently_closed: Dict[str, dict] = {}
        # quality gate 日志节流：{(symbol, side, reason): last_log_ts}，避免每tick刷屏
        self._qgate_log_ts: Dict[tuple, float] = {}
        # alpha_flip 确认计时：{symbol: first_flip_ts}，连续 N 秒反向才平仓
        self._flip_confirm_ts: Dict[str, float] = {}
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
        if sym in self._symbol_blacklist:
            return

        price = d["price"]
        self.feature_engine.on_trade(
            symbol         = sym,
            price          = price,
            qty            = d["qty"],
            is_buyer_maker = d["is_buyer_maker"],
            ts_ms          = d["timestamp"],
        )
        self._tick_count += 1

        # ── 价格轨迹采样（每1秒一个点，仅对持仓品种）────────────────────────────
        if sym in self._price_trails:
            _trail = self._price_trails[sym]
            _now_t = time.time()
            if _now_t - _trail["last_record_ts"] >= 1.0:
                _trail["points"].append((_now_t, price))
                _trail["last_record_ts"] = _now_t

        # ── 插针检测 L1（成交维度）────────────────────────────────────────────
        if self.params["shock_detector_enabled"]:
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

        # ════════════════════════════════════════════════════════════════════
        # 四层退出体系（Lifecycle 驱动）
        # Layer 1 → vol 硬止损（绝对底线，防爆仓）
        # Layer 2 → REVERSAL 主平仓（alpha 逻辑翻转，慢层每 rank_interval 触发）
        # Layer 3 → Lifecycle 状态感知 Trailing（DECAY 收紧，EXPANSION 宽松）
        # Layer 4 → 时间退出（持仓过久且无盈利）
        # ════════════════════════════════════════════════════════════════════
        _now_exit = time.time()
        _min_hold = self.params["min_hold_seconds"]

        # ── Layer 1：alpha 置信度驱动的动态止损 ─────────────────────────────
        # sl_dist = open_vol × (base_mult + conf_mult × confidence)
        # confidence ∈ [0,1]：强 alpha → 止损更宽（给空间），弱 alpha → 止损更紧（快速止损）
        # 同时检测低置信度持续偏低 + 不利浮亏 → low_confidence_cut 提前退出
        if not self.params.get("hold_forever", False):
            _base_mult   = self.params.get("sl_base_mult", 2.0)
            _conf_mult   = self.params.get("sl_conf_mult", 3.0)
            _lc_thresh   = self.params.get("low_conf_threshold", 0.25)
            _adv_loss    = self.params.get("low_conf_adverse_loss", 0.005)
            _lc_rounds   = self.params.get("low_conf_rounds", 4)

            if sym in self.long_positions and sym not in self.short_positions:
                pos      = self.long_positions[sym]
                entry    = pos["entry_price"]
                open_vol = pos.get("open_vol", self._calc_volatility(sym))
                _lc      = self.lifecycle_tracker.get_lifecycle(sym, "LONG")
                conf     = self._calc_confidence(_lc, pos)
                sl_dist  = open_vol * (_base_mult + _conf_mult * conf)

                # ── dynamic_vol_stop ───────────────────────────────────────
                if entry > 0 and price < entry * (1 - sl_dist):
                    _tid = pos.get("trade_id", "")
                    cur_ret = (price - entry) / entry
                    logger.warning(
                        f"[Exit-L1] {sym} LONG dynamic_vol_stop "
                        f"entry={entry:.4f} price={price:.4f} "
                        f"sl_dist={sl_dist:.3%} conf={conf:.3f} open_vol={open_vol:.4%}"
                    )
                    self.trade_recorder.record_rank_snapshot(
                        trade_id=_tid,
                        aligned_score=_lc._aligned_score() if _lc else 0.0,
                        velocity=_lc.velocity if _lc else 0.0,
                        lc_state=_lc.state.value if _lc else "",
                        price=price,
                        extra={"confidence": round(conf, 4), "sl_dist": round(sl_dist, 6), "exit_trigger": "dynamic_vol_stop"},
                    )
                    self._close_long(sym, "dynamic_vol_stop")

                # ── low_confidence_cut ─────────────────────────────────────
                elif entry > 0:
                    cur_ret = (price - entry) / entry
                    if conf < _lc_thresh:
                        pos["_conf_low_count"] = pos.get("_conf_low_count", 0) + 1
                    else:
                        pos["_conf_low_count"] = 0
                    if (pos.get("_conf_low_count", 0) >= _lc_rounds
                            and cur_ret < -_adv_loss):
                        _tid = pos.get("trade_id", "")
                        logger.warning(
                            f"[Exit-L1] {sym} LONG low_confidence_cut "
                            f"conf={conf:.3f} <{_lc_thresh} × {_lc_rounds}轮 "
                            f"ret={cur_ret:.3%} < -{_adv_loss:.3%}"
                        )
                        self.trade_recorder.record_rank_snapshot(
                            trade_id=_tid,
                            aligned_score=_lc._aligned_score() if _lc else 0.0,
                            velocity=_lc.velocity if _lc else 0.0,
                            lc_state=_lc.state.value if _lc else "",
                            price=price,
                            extra={"confidence": round(conf, 4), "conf_low_count": pos["_conf_low_count"], "cur_ret": round(cur_ret, 6), "exit_trigger": "low_confidence_cut"},
                        )
                        self._close_long(sym, "low_confidence_cut")

            elif sym in self.short_positions and sym not in self.long_positions:
                pos      = self.short_positions[sym]
                entry    = pos["entry_price"]
                open_vol = pos.get("open_vol", self._calc_volatility(sym))
                _lc      = self.lifecycle_tracker.get_lifecycle(sym, "SHORT")
                conf     = self._calc_confidence(_lc, pos)
                sl_dist  = open_vol * (_base_mult + _conf_mult * conf)

                # ── dynamic_vol_stop ───────────────────────────────────────
                if entry > 0 and price > entry * (1 + sl_dist):
                    _tid = pos.get("trade_id", "")
                    cur_ret = (entry - price) / entry
                    logger.warning(
                        f"[Exit-L1] {sym} SHORT dynamic_vol_stop "
                        f"entry={entry:.4f} price={price:.4f} "
                        f"sl_dist={sl_dist:.3%} conf={conf:.3f} open_vol={open_vol:.4%}"
                    )
                    self.trade_recorder.record_rank_snapshot(
                        trade_id=_tid,
                        aligned_score=_lc._aligned_score() if _lc else 0.0,
                        velocity=_lc.velocity if _lc else 0.0,
                        lc_state=_lc.state.value if _lc else "",
                        price=price,
                        extra={"confidence": round(conf, 4), "sl_dist": round(sl_dist, 6), "exit_trigger": "dynamic_vol_stop"},
                    )
                    self._close_short(sym, "dynamic_vol_stop")

                # ── low_confidence_cut ─────────────────────────────────────
                elif entry > 0:
                    cur_ret = (entry - price) / entry
                    if conf < _lc_thresh:
                        pos["_conf_low_count"] = pos.get("_conf_low_count", 0) + 1
                    else:
                        pos["_conf_low_count"] = 0
                    if (pos.get("_conf_low_count", 0) >= _lc_rounds
                            and cur_ret < -_adv_loss):
                        _tid = pos.get("trade_id", "")
                        logger.warning(
                            f"[Exit-L1] {sym} SHORT low_confidence_cut "
                            f"conf={conf:.3f} <{_lc_thresh} × {_lc_rounds}轮 "
                            f"ret={cur_ret:.3%} < -{_adv_loss:.3%}"
                        )
                        self.trade_recorder.record_rank_snapshot(
                            trade_id=_tid,
                            aligned_score=_lc._aligned_score() if _lc else 0.0,
                            velocity=_lc.velocity if _lc else 0.0,
                            lc_state=_lc.state.value if _lc else "",
                            price=price,
                            extra={"confidence": round(conf, 4), "conf_low_count": pos["_conf_low_count"], "cur_ret": round(cur_ret, 6), "exit_trigger": "low_confidence_cut"},
                        )
                        self._close_short(sym, "low_confidence_cut")

        # ── Layer 3：Lifecycle 状态感知 Trailing（仅在有盈利时激活）────────────
        # 峰值价格追踪：每 tick 更新最高/最低价
        _trail_min_profit = self.params["trailing_min_profit"]
        _exp_ratio  = self.params["trailing_expansion_ratio"]
        _decay_ratio = self.params["trailing_decay_ratio"]

        if sym in self.long_positions:
            pos = self.long_positions[sym]
            entry = pos["entry_price"]
            # 更新峰值价格
            if price > pos.get("max_price", entry):
                pos["max_price"] = price
            # 更新 peak_alpha（供 lifecycle 参考）
            _fa_tick = self._latest_fused_alphas.get(sym)
            if _fa_tick is not None and _fa_tick.unified > pos.get("peak_alpha", 0.0):
                pos["peak_alpha"] = _fa_tick.unified
            # trailing 触发检查
            if entry > 0:
                cur_ret  = (price - entry) / entry
                max_p    = pos.get("max_price", entry)
                drawdown = (max_p - price) / max_p if max_p > 0 else 0.0
                vol      = pos.get("open_vol", self._calc_volatility(sym))
                _lc      = self.lifecycle_tracker.get_lifecycle(sym, "LONG")
                _lc_state = _lc.state if _lc else None
                held     = _now_exit - pos["entry_time"]
                _tid     = pos.get("trade_id", "")
                # trailing_armed：首次超过盈利阈值时记录（用 flag 避免重复）
                if cur_ret >= _trail_min_profit and not pos.get("_trailing_armed"):
                    pos["_trailing_armed"] = True
                    self.trade_recorder.record_trailing_armed(
                        trade_id=_tid, price=price, cur_ret=cur_ret,
                        lc_state=_lc_state.value if _lc_state else "",
                    )
                if cur_ret >= _trail_min_profit and held >= _min_hold:
                    if _lc_state == AlphaState.DECAY:
                        thresh = vol * _decay_ratio
                        if drawdown > thresh:
                            self.trade_recorder.record_trailing_hit(
                                trade_id=_tid, price=price, cur_ret=cur_ret,
                                drawdown=drawdown,
                                lc_state=_lc_state.value if _lc_state else "",
                                threshold=thresh,
                            )
                            logger.info(
                                f"[Exit-L3] {sym} LONG trailing_decay "
                                f"drawdown={drawdown:.3%} > thresh={thresh:.3%} "
                                f"(vol={vol:.4%}×{_decay_ratio})"
                            )
                            self._close_long(sym, "trailing_decay")
                    elif _lc_state == AlphaState.EXPANSION:
                        thresh = vol * _exp_ratio
                        if drawdown > thresh:
                            self.trade_recorder.record_trailing_hit(
                                trade_id=_tid, price=price, cur_ret=cur_ret,
                                drawdown=drawdown,
                                lc_state=_lc_state.value if _lc_state else "",
                                threshold=thresh,
                            )
                            logger.info(
                                f"[Exit-L3] {sym} LONG trailing_expansion "
                                f"drawdown={drawdown:.3%} > thresh={thresh:.3%} "
                                f"(vol={vol:.4%}×{_exp_ratio})"
                            )
                            self._close_long(sym, "trailing_expansion")

        elif sym in self.short_positions:
            pos = self.short_positions[sym]
            entry = pos["entry_price"]
            if price < pos.get("min_price", entry):
                pos["min_price"] = price
            _fa_tick = self._latest_fused_alphas.get(sym)
            if _fa_tick is not None and (-_fa_tick.unified) > pos.get("peak_alpha", 0.0):
                pos["peak_alpha"] = -_fa_tick.unified
            if entry > 0:
                cur_ret  = (entry - price) / entry
                min_p    = pos.get("min_price", entry)
                drawdown = (price - min_p) / min_p if min_p > 0 else 0.0
                vol      = pos.get("open_vol", self._calc_volatility(sym))
                _lc      = self.lifecycle_tracker.get_lifecycle(sym, "SHORT")
                _lc_state = _lc.state if _lc else None
                held     = _now_exit - pos["entry_time"]
                _tid     = pos.get("trade_id", "")
                if cur_ret >= _trail_min_profit and not pos.get("_trailing_armed"):
                    pos["_trailing_armed"] = True
                    self.trade_recorder.record_trailing_armed(
                        trade_id=_tid, price=price, cur_ret=cur_ret,
                        lc_state=_lc_state.value if _lc_state else "",
                    )
                if cur_ret >= _trail_min_profit and held >= _min_hold:
                    if _lc_state == AlphaState.DECAY:
                        thresh = vol * _decay_ratio
                        if drawdown > thresh:
                            self.trade_recorder.record_trailing_hit(
                                trade_id=_tid, price=price, cur_ret=cur_ret,
                                drawdown=drawdown,
                                lc_state=_lc_state.value if _lc_state else "",
                                threshold=thresh,
                            )
                            logger.info(
                                f"[Exit-L3] {sym} SHORT trailing_decay "
                                f"drawdown={drawdown:.3%} > thresh={thresh:.3%} "
                                f"(vol={vol:.4%}×{_decay_ratio})"
                            )
                            self._close_short(sym, "trailing_decay")
                    elif _lc_state == AlphaState.EXPANSION:
                        thresh = vol * _exp_ratio
                        if drawdown > thresh:
                            self.trade_recorder.record_trailing_hit(
                                trade_id=_tid, price=price, cur_ret=cur_ret,
                                drawdown=drawdown,
                                lc_state=_lc_state.value if _lc_state else "",
                                threshold=thresh,
                            )
                            logger.info(
                                f"[Exit-L3] {sym} SHORT trailing_expansion "
                                f"drawdown={drawdown:.3%} > thresh={thresh:.3%} "
                                f"(vol={vol:.4%}×{_exp_ratio})"
                            )
                            self._close_short(sym, "trailing_expansion")

        # ── Layer 4：时间退出（持仓超限且未盈利）────────────────────────────────
        _max_hold  = self.params.get("max_hold_seconds", 3600)
        _te_min_profit = self.params.get("time_exit_min_profit", 0.001)
        if sym in self.long_positions:
            pos   = self.long_positions[sym]
            held  = _now_exit - pos["entry_time"]
            entry = pos["entry_price"]
            ret   = (price - entry) / entry if entry > 0 else 0.0
            if held >= _max_hold and ret < _te_min_profit:
                logger.info(
                    f"[Exit-L4] {sym} LONG time_exit "
                    f"held={held/3600:.1f}h ret={ret:.3%} < {_te_min_profit:.3%}"
                )
                self._close_long(sym, "time_exit")
        elif sym in self.short_positions:
            pos   = self.short_positions[sym]
            held  = _now_exit - pos["entry_time"]
            entry = pos["entry_price"]
            ret   = (entry - price) / entry if entry > 0 else 0.0
            if held >= _max_hold and ret < _te_min_profit:
                logger.info(
                    f"[Exit-L4] {sym} SHORT time_exit "
                    f"held={held/3600:.1f}h ret={ret:.3%} < {_te_min_profit:.3%}"
                )
                self._close_short(sym, "time_exit")

        # ── LOB Timing：实时开仓触发（alpha_flip 平仓已移除，由 Layer 2 REVERSAL 替代）
        feat = self._latest_features.get(sym)
        if feat is not None:
            now_t = time.time()

            # 实时开仓：候选池内 + timing_score 超阈值 + 未持仓 + 仓位未满
            # 使用 _effective_timing_threshold() 支持无交易时放宽模式
            timing_score = self.timing_engine.get_timing_score(sym, feat)
            eff_thresh   = self._effective_timing_threshold()

            if (sym in self._candidate_pool["long"]
                    and sym not in self.long_positions
                    and sym not in self.short_positions
                    and len(self.long_positions) < self.params["max_long_positions"]
                    and not (self.params["kill_switch_enabled"] and self.shock_detector.is_kill_switched())
                    and not self.timing_engine._cooling_down(sym)
                    and timing_score > eff_thresh):
                regime = self._regime_filter(self._latest_features)
                if regime["allow_long"] and self._pass_quality_gate(sym, feat, "LONG"):
                    score = self._latest_scores.get(sym, 0.0)
                    self._open_long(sym, score, price, self._latest_features)
                    self.timing_engine.record_entry(sym)

            elif (sym in self._candidate_pool["short"]
                    and sym not in self.short_positions
                    and sym not in self.long_positions
                    and len(self.short_positions) < self.params["max_short_positions"]
                    and not (self.params["kill_switch_enabled"] and self.shock_detector.is_kill_switched())
                    and not self.timing_engine._cooling_down(sym)
                    and timing_score < -eff_thresh):
                regime = self._regime_filter(self._latest_features)
                if regime["allow_short"] and self._pass_quality_gate(sym, feat, "SHORT"):
                    score = self._latest_scores.get(sym, 0.0)
                    self._open_short(sym, score, price, self._latest_features)
                    self.timing_engine.record_entry(sym)

        now = time.time()
        # _external_rank_loop=True 时由外部线程定时调用 _rank_and_trade，此处跳过
        if not self._external_rank_loop and now - self._last_rank_time >= self.params["rank_interval"]:
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
        if self.params["shock_detector_enabled"]:
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
            pc1    = self.lob_engine.get_symbol_pc1(sym)
            self.feature_engine.update_lob_latent(sym, z, bucket, pc1=pc1)

    def update_derivatives(self, symbol: str, funding_rate: float, oi: float, ret_24h: float = 0.0):
        self.feature_engine.update_derivatives(symbol, funding_rate, oi, ret_24h)

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

    def _calc_confidence(self, lc, pos: dict) -> float:
        """
        计算 alpha 置信度 ∈ [0, 1]，用于动态止损距离调整。

        组成：
            - aligned_score  (主项，权重 0.6)：方向对齐后的 alpha 强度
            - velocity       (辅项，权重 0.2)：score 变化速度（正=增强，负=衰减）
            - lifecycle_bonus (辅项，权重 0.2)：状态奖罚
                EXPANSION → +0.2, BUILD → 0.0, DECAY → -0.2, REVERSAL → -0.5
        最终结果 clamp 到 [0, 1]。
        """
        if lc is None:
            # 无 lifecycle 数据（热身期）：取 entry_score 做保守估算
            score_raw = pos.get("entry_score", 0.5)
            return max(0.0, min(1.0, score_raw))

        aligned = lc._aligned_score()    # LONG→score，SHORT→-score；方向正向为正
        vel     = lc._aligned_vel()      # 同上方向对齐

        # aligned_score 贡献：强度从 [0, 1] 线性映射（分数本身已约束在 [-1,1] 区间）
        score_contrib = max(0.0, min(1.0, aligned))

        # velocity 贡献：正速度 +加分，负速度 +减分，映射到 [-0.5, +0.5] 再归一化
        vel_contrib = max(-0.5, min(0.5, vel * 2.0))  # 假设 vel 有效范围约 ±0.25

        # lifecycle 状态奖罚
        state = lc.state
        if state == AlphaState.EXPANSION:
            lc_bonus = 0.2
        elif state == AlphaState.BUILD:
            lc_bonus = 0.0
        elif state == AlphaState.DECAY:
            lc_bonus = -0.2
        else:  # REVERSAL
            lc_bonus = -0.5

        raw = 0.6 * score_contrib + 0.2 * vel_contrib + 0.2 * lc_bonus
        return max(0.0, min(1.0, raw + 0.5))  # 偏置 +0.5 使中性 alpha 也有 50% 置信度

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

    def _save_price_trail(self, symbol: str, exit_price: float, reason: str):
        """平仓时将价格轨迹写入 CSV 文件（供 view_price_trail.py 可视化）"""
        trail = self._price_trails.pop(symbol, None)
        if trail is None or not trail["points"]:
            return
        try:
            entry_ts  = int(trail["entry_time"])
            fname     = f"{symbol}_{trail['side']}_{entry_ts}.csv"
            fpath     = os.path.join(self._trail_dir, fname)
            lev       = self.params["leverage"]
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    f"# symbol={symbol}",
                    f"side={trail['side']}",
                    f"entry_price={trail['entry_price']}",
                    f"exit_price={exit_price}",
                    f"exit_reason={reason}",
                    f"sl_price={trail['sl_price']}",
                    f"tp_price={trail['tp_price']}",
                    f"leverage={lev}",
                ])
                w.writerow(["ts", "seconds", "price", "ret_margin_pct"])
                entry_p = trail["entry_price"]
                entry_t = trail["points"][0][0]
                for ts, p in trail["points"]:
                    secs = round(ts - entry_t, 1)
                    if trail["side"] == "LONG":
                        ret_m = (p - entry_p) / entry_p * lev * 100
                    else:
                        ret_m = (entry_p - p) / entry_p * lev * 100
                    w.writerow([round(ts, 3), secs, p, round(ret_m, 4)])
                # 写入平仓点
                exit_t = time.time()
                if trail["side"] == "LONG":
                    exit_ret_m = (exit_price - entry_p) / entry_p * lev * 100 if entry_p > 0 else 0
                else:
                    exit_ret_m = (entry_p - exit_price) / entry_p * lev * 100 if entry_p > 0 else 0
                w.writerow([round(exit_t, 3), round(exit_t - entry_t, 1), exit_price, round(exit_ret_m, 4)])
        except Exception as e:
            logger.debug(f"[AlphaFactory] price_trail 写入失败 {symbol}: {e}")

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

    # ─── 防锁死：放宽模式辅助方法 ────────────────────────────────────────────

    def _is_relax_mode(self) -> bool:
        """连续无交易轮数超过阈值时进入放宽模式。"""
        return self._no_trade_rounds >= self.params["no_trade_relax_rounds"]

    def _effective_timing_threshold(self) -> float:
        """放宽模式下降低 LOB timing 入场阈值。"""
        base = self.timing_engine.entry_threshold
        if self._is_relax_mode():
            return base * self.params["relax_timing_factor"]
        return base

    def _effective_min_edge(self) -> float:
        """放宽模式下降低成本覆盖要求。"""
        base = self.params["min_edge_multiple"]
        if self._is_relax_mode():
            return base * self.params["relax_edge_factor"]
        return base

    def _effective_min_vol_zscore(self) -> float:
        """放宽模式下降低成交量门槛。"""
        base = self.params["min_volume_zscore"]
        if self._is_relax_mode():
            return base * self.params["relax_min_vol_factor"]
        return base

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

        # ── 1. 市场状态引擎（终极版）────────────────────────────────────────
        market_state = self.market_state_engine.update(features)
        self._latest_market_state = market_state

        if not market_state.is_tradeable:
            logger.info(
                f"[MarketState] 市场不可交易 regime={market_state.regime.value} "
                f"dispersion={market_state.dispersion:.4f} "
                f"tradability={market_state.tradability:.3f}，跳过本轮"
            )
            # 市场不可交易时仍然更新持仓生命周期（可能需要平仓）
            self._update_lifecycle_states({})
            self._log_positions()
            return

        # 旧版市场活跃度过滤（与 MarketStateEngine 互补，保留向后兼容）
        if not self._is_market_tradeable(features):
            self._update_lifecycle_states({})
            return

        # 过滤运行时黑名单（无效合约）
        if self._symbol_blacklist:
            features = {s: f for s, f in features.items() if s not in self._symbol_blacklist}

        # 将市场制度同步给 ScoringEngine，使动量因子方向自适应
        self.scoring_engine.set_regime(market_state.regime.value)

        scores = self.scoring_engine.compute_scores(features)
        if not scores:
            return

        # Dispersion Filter（MarketStateEngine 已包含，此处保留作双重保护）
        score_vals = list(scores.values())
        n_s = len(score_vals)
        if n_s > 1:
            mean_s = sum(score_vals) / n_s
            disp   = (sum((v - mean_s) ** 2 for v in score_vals) / n_s) ** 0.5
            if disp < self.params["min_score_dispersion"]:
                logger.info(f"[AlphaFactory] 截面分散度不足 std={disp:.3f}，跳过候选池更新")
                self._update_lifecycle_states(self._latest_fused_alphas)
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

        # ── depth5 预热订阅：排序完成后立即订阅（热身期也执行，确保 LOB 数据就绪）
        if self._depth_update_fn is not None:
            self._depth_update_fn(
                set(self.long_positions.keys())
                | set(self.short_positions.keys())
                | {e.symbol for e in long_top}
                | {e.symbol for e in short_top}
            )

        # 热身期：让 EMA 和候选池先稳定
        if self.ranking_engine.rank_count <= self.params["warmup_count"]:
            logger.info(
                f"[AlphaFactory] 热身期 "
                f"({self.ranking_engine.rank_count}/{self.params['warmup_count']})"
            )
            return

        # Kill Switch：冲击过多时暂停新开仓（kill_switch_enabled=False 时完全跳过）
        if self.params["kill_switch_enabled"] and self.shock_detector.is_kill_switched():
            status = self.shock_detector.get_status()
            logger.warning(
                f"[ShockDetector] Kill Switch 触发，暂停新开仓 "
                f"(resume_in={status['kill_resume_in']:.0f}s) "
                f"候选池保留，{status['kill_resume_in']:.0f}s 后自动恢复"
            )
            # 继续执行后续逻辑（更新候选池、生命周期），只是 on_tick 不会实际开仓

        # ── 2. Alpha 融合（终极版）──────────────────────────────────────────
        fused = self.alpha_fusion.fuse(
            scores        = scores,
            features      = features,
            timing_engine = self.timing_engine,
            market_state  = market_state,
        )
        self._latest_fused_alphas = fused

        # ── 3. 更新所有持仓的 alpha 生命周期状态（终极版）───────────────────
        self._update_lifecycle_states(fused)

        # 相关性去重（原有逻辑保留）
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

        # ── 防锁死：更新无交易轮数计数器 ────────────────────────────────────
        since_last = now - self._last_open_time
        rank_iv    = self.params["rank_interval"]
        self._no_trade_rounds = max(0, int(since_last / rank_iv) - 1) if self._last_open_time > 0 else 0
        if self._is_relax_mode():
            logger.info(
                f"[AlphaFactory] 放宽模式 no_trade_rounds={self._no_trade_rounds} "
                f"timing_thresh={self._effective_timing_threshold():.2f} "
                f"min_edge={self._effective_min_edge():.2f}x"
            )

        # ── 候选池开仓分数明细（每轮排序打印，便于诊断为何没有交易）────────────
        self._log_candidate_scores(market_state)

        self._log_positions()
        self._export_positions_snapshot()

    def _update_lifecycle_states(self, fused: dict):
        """
        Layer 2（慢层，每 rank_interval 秒）：
            - 更新所有持仓的 AlphaLifecycle 状态机
            - REVERSAL → 立即主平仓（exit_reason=lifecycle_reversal）
            - DECAY    → 记录日志，通知快层（on_tick）收紧 trailing 阈值
            - BUILD/EXPANSION → 维持正常 trailing 宽度
        """
        if not self.params.get("lifecycle_exit", True):
            return

        min_hold = self.params["min_hold_seconds"]
        now      = time.time()

        def _process(sym, side, positions, aligned_score_fn, close_fn):
            fa = fused.get(sym)
            unified_score = fa.unified if fa else self._latest_scores.get(sym, 0.0)
            old_lc    = self.lifecycle_tracker.get_lifecycle(sym, side)
            old_state = old_lc.state if old_lc else None
            new_state = self.lifecycle_tracker.update(sym, side, unified_score)
            new_lc    = self.lifecycle_tracker.get_lifecycle(sym, side)
            pos       = positions.get(sym)
            aligned   = aligned_score_fn(unified_score)
            velocity  = new_lc.velocity if new_lc else 0.0
            if pos:
                if aligned > pos.get("peak_alpha", 0.0):
                    pos["peak_alpha"] = aligned
            held  = now - positions[sym]["entry_time"]
            price = self._get_price(sym)
            cur_ret_val = (
                (price - pos["entry_price"]) / pos["entry_price"]
                if side == "LONG" and pos and pos["entry_price"] > 0
                else (pos["entry_price"] - price) / pos["entry_price"]
                if pos and pos["entry_price"] > 0 else 0.0
            )
            trade_id = pos.get("trade_id", "") if pos else ""
            # rank_snapshot：每轮记录 alpha 快照
            self.trade_recorder.record_rank_snapshot(
                trade_id      = trade_id,
                aligned_score = aligned,
                velocity      = velocity,
                lc_state      = new_state.value,
                price         = price,
                extra         = f"unified={unified_score:+.3f} held={held:.0f}s",
            )
            # 状态变化：日志 + TradeRecorder
            if new_state != old_state:
                logger.info(
                    f"[Exit-L2] {sym} {side} lifecycle {old_state.value if old_state else '?'}"
                    f" → {new_state.value} (unified={unified_score:+.3f} held={held:.0f}s)"
                )
                self.trade_recorder.record_lifecycle_change(
                    trade_id      = trade_id,
                    old_state     = old_state.value if old_state else "",
                    new_state     = new_state.value,
                    aligned_score = aligned,
                    velocity      = velocity,
                    cur_ret       = cur_ret_val,
                    price         = price,
                )
            # REVERSAL → 主平仓
            if new_state == AlphaState.REVERSAL and held >= min_hold:
                close_fn(sym, "lifecycle_reversal")

            # ── DECAY 提前止错：confidence 持续偏低 + 负动量 → early_reversal ──
            # 不改变状态机本身，也不干扰 dynamic_vol_stop / trailing，
            # 仅在 DECAY 阶段 confidence < 阈值且 aligned_vel < 0 连续 N 轮时主动退出，
            # 避免等待完整 REVERSAL 转换期间亏损扩大。
            _er_conf_thresh = self.params.get("early_reversal_conf_thresh", 0.30)
            _er_rounds      = self.params.get("early_reversal_rounds", 3)
            if new_state == AlphaState.DECAY and held >= min_hold and pos:
                conf        = self._calc_confidence(new_lc, pos)
                aligned_vel = new_lc._aligned_vel() if new_lc else 0.0
                if conf < _er_conf_thresh and aligned_vel < 0:
                    pos["_decay_early_exit_count"] = pos.get("_decay_early_exit_count", 0) + 1
                else:
                    pos["_decay_early_exit_count"] = 0
                if pos.get("_decay_early_exit_count", 0) >= _er_rounds:
                    logger.warning(
                        f"[Exit-L2] {sym} {side} early_reversal "
                        f"DECAY×{pos['_decay_early_exit_count']}轮 "
                        f"conf={conf:.3f}<{_er_conf_thresh} vel={aligned_vel:+.4f}<0 "
                        f"ret={cur_ret_val:.3%}"
                    )
                    self.trade_recorder.record_rank_snapshot(
                        trade_id      = trade_id,
                        aligned_score = aligned,
                        velocity      = velocity,
                        lc_state      = new_state.value,
                        price         = price,
                        extra         = {
                            "confidence":       round(conf, 4),
                            "decay_exit_count": pos["_decay_early_exit_count"],
                            "cur_ret":          round(cur_ret_val, 6),
                            "exit_trigger":     "early_reversal",
                        },
                    )
                    close_fn(sym, "early_reversal")
            elif pos:
                pos["_decay_early_exit_count"] = 0

        for sym in list(self.long_positions.keys()):
            _process(sym, "LONG", self.long_positions,
                     lambda u: u,
                     self._close_long)

        for sym in list(self.short_positions.keys()):
            _process(sym, "SHORT", self.short_positions,
                     lambda u: -u,
                     self._close_short)

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

        # ── 成本覆盖检查（终极版，支持放宽模式动态阈值）────────────────────────
        feat_snap = self._latest_features.get(symbol)
        fa        = self._latest_fused_alphas.get(symbol)
        unified   = fa.unified if fa else score
        if feat_snap is not None:
            cost_est  = self.cost_model.estimate_from_features(symbol, trade_usdt, unified, feat_snap)
            eff_edge  = self._effective_min_edge()
            is_viable = cost_est.expected_gross >= eff_edge * cost_est.total_cost
            if not is_viable:
                logger.debug(
                    f"[CostModel] {symbol} LONG 跳过：{cost_est.reject_reason}"
                    + (" [relax_mode]" if self._is_relax_mode() else "")
                )
                return

        # Layer 1 硬止损距离：入场时波动率 × sl_vol_mult（锁定，不随行情更新）
        vol      = self._calc_volatility(symbol)
        sl_dist  = vol * self.params["sl_vol_mult"]
        sl_price_vol = price * (1 - sl_dist)

        result = self.market_buy(symbol, qty, reduce_only=False)
        if result is not None:
            # 防锁死：重置无交易计数器
            self._last_open_time  = time.time()
            self._no_trade_rounds = 0
            # 注册 alpha 生命周期 + TradeRecorder
            self.lifecycle_tracker.open_position(symbol, "LONG", unified)
            _trade_id = self.trade_recorder.open(
                symbol=symbol, side="LONG", price=price,
                qty=qty, leverage=self.params["leverage"],
                open_vol=vol, entry_score=unified,
            )
            factors = self.scoring_engine.get_factor_breakdown(symbol, features) if features else {}
            factors = dict(factors or {})
            # 快层（高频）因子：hf_ 前缀，与慢层区分
            _fs = self._latest_features.get(symbol)
            _fa = self._latest_fused_alphas.get(symbol)
            if _fs is not None:
                # ofi 归一化：乘以 last_price 转为 USDT 名义值，再用 tanh 压缩到 [-1, +1]
                _ofi_usdt = _fs.ofi * _fs.last_price if _fs.last_price > 1e-10 else _fs.ofi
                _ofi_norm = math.tanh(_ofi_usdt / 1000.0)  # 1000 USDT 为缩放基准
                factors["hf_ofi"]      = round(_ofi_norm,    4)
                factors["hf_lob_pc1"]  = round(_fs.lob_pc1,  4)
                factors["hf_lob_z1"]   = round(_fs.lob_z1,   4)
                factors["hf_lob_z2"]   = round(_fs.lob_z2,   4)
            if _fa is not None:
                factors["hf_fast_boost"]   = round(getattr(_fa, "fast_boost_val", 0.0), 4)
                _ts = self.timing_engine.get_timing_score(symbol, _fs) if _fs else 0.0
                factors["hf_timing_score"] = round(_ts, 4)
            factors["hf_microprice"] = round(self.timing_engine._microprice_zscore(symbol), 4)
            self.long_positions[symbol] = {
                "entry_price": price,
                "qty":         qty,
                "entry_time":  time.time(),
                "score":       score,
                "factors":     factors,
                "max_price":   price,      # Layer 3 trailing 用：持仓期间最高价
                "open_vol":    vol,        # Layer 1 硬止损用：入场时锁定波动率（不随行情更新）
                "entry_alpha": unified,    # 开仓时 unified alpha
                "peak_alpha":  unified,    # 持仓期间最大 unified alpha（方向对齐）
                "trade_id":    _trade_id,  # TradeRecorder 关联 key
            }
            sl_price = sl_price_vol
            if not self.params.get("hold_forever", False):
                sl_res = self.stop_order(symbol, "SELL", "STOP_MARKET", sl_price)
                self._tp_sl_orders[symbol] = {
                    "sl_id":    sl_res.get("orderId") if sl_res else None,
                    "tp_id":    None,
                    "sl_price": sl_price,
                    "tp_price": None,
                }
            # 价格轨迹录制启动
            tp_price = price * (1 + self.params.get("take_profit_pct", 0.0)) if self.params.get("take_profit_pct", 0.0) > 0 else None
            self._price_trails[symbol] = {
                "side":           "LONG",
                "entry_price":    price,
                "entry_time":     time.time(),
                "sl_price":       sl_price,
                "tp_price":       tp_price,
                "points":         [(time.time(), price)],
                "last_record_ts": time.time(),
            }
            fa_snap   = self._latest_fused_alphas.get(symbol)
            ms        = self._latest_market_state
            regime_s  = ms.regime.value if ms else "?"
            trad_s    = f"{ms.tradability:.2f}" if ms else "?"
            vol_z_s   = f"{feat_snap.volume_zscore:+.2f}" if feat_snap else "?"
            spread_s  = f"{feat_snap.spread_bps:.1f}" if feat_snap else "?"
            timing_s  = f"{fa_snap.fast_boost_val:+.3f}" if fa_snap else "?"
            # 因子分解（简化，直接从 features 拿原始值）
            r1m_s = f"{feat_snap.ret_1m:+.4f}" if feat_snap else "?"
            r5m_s = f"{feat_snap.ret_5m:+.4f}" if feat_snap else "?"
            oi_s  = f"{feat_snap.oi_change_pct:+.4f}" if feat_snap else "?"
            fund_s = f"{feat_snap.funding_rate:+.6f}" if feat_snap else "?"
            cost_s = (
                f"fee={cost_est.fee_cost:.3f} spd={cost_est.spread_cost:.3f} "
                f"imp={cost_est.impact_cost:.3f} tot={cost_est.total_cost:.3f} "
                f"exp={cost_est.expected_gross:.3f} ratio={cost_est.expected_gross/(cost_est.total_cost+1e-10):.2f}x"
            ) if feat_snap else "cost=N/A"
            logger.info(
                f"[AlphaFactory] ▲ OPEN LONG  {symbol} | "
                f"price={price:.4f} qty={qty:.4f}({size_scale:.0%}) | "
                f"score={score:+.3f} unified={unified:+.3f} timing={timing_s} | "
                f"regime={regime_s} trad={trad_s} vol_z={vol_z_s} spread={spread_s}bps | "
                f"ret1m={r1m_s} ret5m={r5m_s} oi={oi_s} fund={fund_s} | "
                f"vol={vol:.4%} SL={sl_price:.4f} | {cost_s}"
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

        # ── 成本覆盖检查（终极版，支持放宽模式动态阈值）────────────────────────
        feat_snap = self._latest_features.get(symbol)
        fa        = self._latest_fused_alphas.get(symbol)
        unified   = fa.unified if fa else score
        if feat_snap is not None:
            cost_est  = self.cost_model.estimate_from_features(symbol, trade_usdt, unified, feat_snap)
            eff_edge  = self._effective_min_edge()
            is_viable = cost_est.expected_gross >= eff_edge * cost_est.total_cost
            if not is_viable:
                logger.debug(
                    f"[CostModel] {symbol} SHORT 跳过：{cost_est.reject_reason}"
                    + (" [relax_mode]" if self._is_relax_mode() else "")
                )
                return

        # Layer 1 硬止损距离：入场时波动率 × sl_vol_mult（锁定，不随行情更新）
        vol      = self._calc_volatility(symbol)
        sl_dist  = vol * self.params["sl_vol_mult"]
        sl_price_vol = price * (1 + sl_dist)

        result = self.market_sell(symbol, qty, reduce_only=False)
        if result is not None:
            # 防锁死：重置无交易计数器
            self._last_open_time  = time.time()
            self._no_trade_rounds = 0
            # 注册 alpha 生命周期 + TradeRecorder
            self.lifecycle_tracker.open_position(symbol, "SHORT", unified)
            _trade_id = self.trade_recorder.open(
                symbol=symbol, side="SHORT", price=price,
                qty=qty, leverage=self.params["leverage"],
                open_vol=vol, entry_score=unified,
            )
            factors = self.scoring_engine.get_factor_breakdown(symbol, features) if features else {}
            factors = dict(factors or {})
            # 快层（高频）因子：hf_ 前缀
            _fs = self._latest_features.get(symbol)
            _fa = self._latest_fused_alphas.get(symbol)
            if _fs is not None:
                # ofi 归一化：乘以 last_price 转为 USDT 名义值，再用 tanh 压缩到 [-1, +1]
                _ofi_usdt = _fs.ofi * _fs.last_price if _fs.last_price > 1e-10 else _fs.ofi
                _ofi_norm = math.tanh(_ofi_usdt / 1000.0)  # 1000 USDT 为缩放基准
                factors["hf_ofi"]      = round(_ofi_norm,    4)
                factors["hf_lob_pc1"]  = round(_fs.lob_pc1,  4)
                factors["hf_lob_z1"]   = round(_fs.lob_z1,   4)
                factors["hf_lob_z2"]   = round(_fs.lob_z2,   4)
            if _fa is not None:
                factors["hf_fast_boost"]   = round(getattr(_fa, "fast_boost_val", 0.0), 4)
                _ts = self.timing_engine.get_timing_score(symbol, _fs) if _fs else 0.0
                factors["hf_timing_score"] = round(_ts, 4)
            factors["hf_microprice"] = round(self.timing_engine._microprice_zscore(symbol), 4)
            self.short_positions[symbol] = {
                "entry_price": price,
                "qty":         qty,
                "entry_time":  time.time(),
                "score":       score,
                "factors":     factors,
                "min_price":   price,      # Layer 3 trailing 用：持仓期间最低价
                "open_vol":    vol,        # Layer 1 硬止损用：入场时锁定波动率（不随行情更新）
                "entry_alpha": unified,    # 开仓时 unified alpha
                "peak_alpha":  unified,    # 持仓期间最大 unified alpha（方向对齐：SHORT 取负）
                "trade_id":    _trade_id,  # TradeRecorder 关联 key
            }
            sl_price = sl_price_vol
            if not self.params.get("hold_forever", False):
                sl_res = self.stop_order(symbol, "BUY", "STOP_MARKET", sl_price)
                self._tp_sl_orders[symbol] = {
                    "sl_id":    sl_res.get("orderId") if sl_res else None,
                    "tp_id":    None,
                    "sl_price": sl_price,
                    "tp_price": None,
                }
            # 价格轨迹录制启动
            tp_price_short = price * (1 - self.params.get("take_profit_pct", 0.0)) if self.params.get("take_profit_pct", 0.0) > 0 else None
            self._price_trails[symbol] = {
                "side":           "SHORT",
                "entry_price":    price,
                "entry_time":     time.time(),
                "sl_price":       sl_price,
                "tp_price":       tp_price_short,
                "points":         [(time.time(), price)],
                "last_record_ts": time.time(),
            }
            fa_snap   = self._latest_fused_alphas.get(symbol)
            ms        = self._latest_market_state
            regime_s  = ms.regime.value if ms else "?"
            trad_s    = f"{ms.tradability:.2f}" if ms else "?"
            vol_z_s   = f"{feat_snap.volume_zscore:+.2f}" if feat_snap else "?"
            spread_s  = f"{feat_snap.spread_bps:.1f}" if feat_snap else "?"
            timing_s  = f"{fa_snap.fast_boost_val:+.3f}" if fa_snap else "?"
            r1m_s  = f"{feat_snap.ret_1m:+.4f}" if feat_snap else "?"
            r5m_s  = f"{feat_snap.ret_5m:+.4f}" if feat_snap else "?"
            oi_s   = f"{feat_snap.oi_change_pct:+.4f}" if feat_snap else "?"
            fund_s = f"{feat_snap.funding_rate:+.6f}" if feat_snap else "?"
            cost_s = (
                f"fee={cost_est.fee_cost:.3f} spd={cost_est.spread_cost:.3f} "
                f"imp={cost_est.impact_cost:.3f} tot={cost_est.total_cost:.3f} "
                f"exp={cost_est.expected_gross:.3f} ratio={cost_est.expected_gross/(cost_est.total_cost+1e-10):.2f}x"
            ) if feat_snap else "cost=N/A"
            logger.info(
                f"[AlphaFactory] ▼ OPEN SHORT {symbol} | "
                f"price={price:.4f} qty={qty:.4f}({size_scale:.0%}) | "
                f"score={score:+.3f} unified={unified:+.3f} timing={timing_s} | "
                f"regime={regime_s} trad={trad_s} vol_z={vol_z_s} spread={spread_s}bps | "
                f"ret1m={r1m_s} ret5m={r5m_s} oi={oi_s} fund={fund_s} | "
                f"vol={vol:.4%} SL={sl_price:.4f} | {cost_s}"
            )

    def _close_long(self, symbol: str, reason: str):
        if self.params.get("hold_forever", False) and reason != "lifecycle_reversal":
            logger.debug(f"[AlphaFactory] hold_forever=True，跳过平仓 LONG {symbol} ({reason})")
            return
        pos = self.long_positions.pop(symbol, None)
        if pos is None:
            return
        self._cancel_tp_sl(symbol)
        self._qgate_fails.pop(symbol, None)
        self._flip_confirm_ts.pop(symbol, None)   # 清理 alpha_flip 确认计时
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

        self._save_price_trail(symbol, exit_price, reason)
        if self._db:
            self._db.save_completed_trade(record)
        # TradeRecorder：平仓归因记录（含 lifecycle 上下文）
        _lc = self.lifecycle_tracker.get_lifecycle(symbol, "LONG")
        _lc_state    = _lc.state.value if _lc else ""
        _aligned_sc  = _lc._aligned_score() if _lc else 0.0
        _velocity    = _lc.velocity if _lc else 0.0
        _peak_pnl_r  = pos.get("peak_pnl", 0.0)
        _max_p       = pos.get("max_price", pos["entry_price"])
        _drawdown    = (_max_p - exit_price) / _max_p if _max_p > 0 else 0.0
        _trade_id    = pos.get("trade_id", "")
        self.trade_recorder.close(
            trade_id     = _trade_id,
            exit_price   = exit_price,
            exit_reason  = reason,
            lc_state     = _lc_state,
            aligned_score = _aligned_sc,
            velocity     = _velocity,
            peak_pnl     = _peak_pnl_r,
            drawdown     = _drawdown,
            cur_ret      = ret,
            pnl_usdt     = pnl_usdt,
            fee_usdt     = fee_usdt,
            ret_pct      = ret * 100,
            ret_lev_pct  = ret_lev * 100,
            hold_seconds = held,
        )
        self.lifecycle_tracker.close_position(symbol, "LONG")
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
        if self.params.get("hold_forever", False) and reason != "lifecycle_reversal":
            logger.debug(f"[AlphaFactory] hold_forever=True，跳过平仓 SHORT {symbol} ({reason})")
            return
        pos = self.short_positions.pop(symbol, None)
        if pos is None:
            return
        self._cancel_tp_sl(symbol)
        self._qgate_fails.pop(symbol, None)
        self._flip_confirm_ts.pop(symbol, None)   # 清理 alpha_flip 确认计时
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

        self._save_price_trail(symbol, exit_price, reason)
        if self._db:
            self._db.save_completed_trade(record)
        # TradeRecorder：平仓归因记录（含 lifecycle 上下文）
        _lc = self.lifecycle_tracker.get_lifecycle(symbol, "SHORT")
        _lc_state    = _lc.state.value if _lc else ""
        _aligned_sc  = _lc._aligned_score() if _lc else 0.0
        _velocity    = _lc.velocity if _lc else 0.0
        _peak_pnl_r  = pos.get("peak_pnl", 0.0)
        _min_p       = pos.get("min_price", pos["entry_price"])
        _drawdown    = (exit_price - _min_p) / _min_p if _min_p > 0 else 0.0
        _trade_id    = pos.get("trade_id", "")
        self.trade_recorder.close(
            trade_id     = _trade_id,
            exit_price   = exit_price,
            exit_reason  = reason,
            lc_state     = _lc_state,
            aligned_score = _aligned_sc,
            velocity     = _velocity,
            peak_pnl     = _peak_pnl_r,
            drawdown     = _drawdown,
            cur_ret      = ret,
            pnl_usdt     = pnl_usdt,
            fee_usdt     = fee_usdt,
            ret_pct      = ret * 100,
            ret_lev_pct  = ret_lev * 100,
            hold_seconds = held,
        )
        self.lifecycle_tracker.close_position(symbol, "SHORT")
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
        if self.params.get("hold_forever", False):
            logger.debug(f"[AlphaFactory] hold_forever=True，跳过交易所侧平仓记录 {side} {symbol}")
            return
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
        if self.params["shock_detector_enabled"] and self.shock_detector.is_paused(symbol):
            logger.debug(f"[AlphaFactory] {symbol} {side} 冲击暂停窗口，跳过")
            return False

        # ── 1. 流动性基础检查（放宽模式下降低量门槛）────────────────────────
        eff_vol_thresh = self._effective_min_vol_zscore()
        if abs(feat.volume_zscore) < eff_vol_thresh:
            _key = (symbol, side, "vol")
            if time.time() - self._qgate_log_ts.get(_key, 0) > 5.0:
                self._qgate_log_ts[_key] = time.time()
                logger.info(
                    f"[AlphaFactory] {symbol} {side} 量不足 "
                    f"|vol_z|={abs(feat.volume_zscore):.2f} (需>|{eff_vol_thresh:.2f}|"
                    + (" [relax]" if self._is_relax_mode() else "") + ")"
                )
            return False
        if feat.spread_bps > self.params["max_spread_bps"]:
            _key = (symbol, side, "spread")
            if time.time() - self._qgate_log_ts.get(_key, 0) > 5.0:
                self._qgate_log_ts[_key] = time.time()
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
        feat = self._latest_features.get(symbol)
        if feat and feat.last_price > 0:
            return feat.last_price
        # 降级：直接从 feature_engine._states 取最新价
        state = self.feature_engine._states.get(symbol)
        if state and state.features.last_price > 0:
            return state.features.last_price
        return 0.0

    def _log_candidate_scores(self, market_state=None):
        """
        每轮排序后打印候选池内每个品种的完整开仓分数，便于诊断为何没有交易。

        输出字段：
            ema      : RankingEngine 的 EMA 慢层得分（进入候选池的依据）
            unified  : AlphaFusion 输出（慢层×regime×tradability×timing×crowding）
            timing   : LOBTimingEngine 实时 timing_score（快层，决定是否真正开仓）
            thresh   : 当前有效 timing 入场阈值（含放宽模式）
            vol_z    : 成交量 Z-score（quality gate 检查项）
            spread   : 买卖价差（bps，quality gate 检查项）
            pass_qg  : 是否通过 quality gate（vol+spread+cooldown）
            cost_ok  : 是否通过成本覆盖检查
        """
        eff_thresh = self._effective_timing_threshold()
        eff_edge   = self._effective_min_edge()
        relax_tag  = " [RELAX]" if self._is_relax_mode() else ""

        for side, pool in (("多头▲", "long"), ("空头▼", "short")):
            syms = list(self._candidate_pool[pool])
            if not syms:
                logger.info(f"[候选池] {side} 空池")
                continue

            lines = []
            for sym in syms:
                fa      = self._latest_fused_alphas.get(sym)
                feat    = self._latest_features.get(sym)
                ema     = self._latest_scores.get(sym, 0.0)
                unified = fa.unified if fa else 0.0

                timing  = self.timing_engine.get_timing_score(sym, feat) if feat else 0.0
                vol_z   = abs(feat.volume_zscore) if feat else 0.0
                spread  = feat.spread_bps if feat else 0.0

                # quality gate 快速评估（不触发冷却期）
                eff_vol = self._effective_min_vol_zscore()
                qg_ok   = (
                    vol_z >= eff_vol
                    and spread <= self.params["max_spread_bps"]
                    and not (self.params["shock_detector_enabled"] and self.shock_detector.is_paused(sym))
                )

                # 成本覆盖快速评估
                cost_ok = True
                if feat is not None:
                    try:
                        cost_est = self.cost_model.estimate_from_features(
                            sym, self.params["trade_size_usdt"], unified, feat
                        )
                        cost_ok = cost_est.is_viable
                    except Exception:
                        pass

                timing_pass = (timing > eff_thresh) if pool == "long" else (timing < -eff_thresh)
                status = "✓READY" if (qg_ok and cost_ok and timing_pass) else (
                    "✗timing" if not timing_pass else
                    "✗qgate"  if not qg_ok else
                    "✗cost"
                )

                lines.append(
                    f"{sym}({status} "
                    f"ema={ema:+.3f} unified={unified:+.3f} "
                    f"timing={timing:+.3f}/±{eff_thresh:.2f} "
                    f"vol={vol_z:.2f} spd={spread:.1f}bps "
                    f"cost={'OK' if cost_ok else 'NO'})"
                )

            regime_str = market_state.regime.value if market_state else "?"
            trad_str   = f"{market_state.tradability:.2f}" if market_state else "?"
            header = (
                f"[候选池] {side}{relax_tag} "
                f"regime={regime_str} tradability={trad_str} "
                f"thresh=±{eff_thresh:.2f} edge={eff_edge:.2f}x"
            )
            logger.info(header)
            for line in lines:
                logger.info(f"  {line}")

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
                "  ── 【慢层·选股因子】盈利能力分析（ScoringEngine，60s更新，权重驱动选股）────",
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

        # ── 高频因子（快层）盈利能力分析 ──────────────────────────────────────
        HF_FACTORS = ["hf_microprice", "hf_ofi", "hf_lob_pc1", "hf_lob_z1", "hf_lob_z2",
                      "hf_fast_boost", "hf_timing_score"]
        hf_trades = [t for t in trades if any(k in t.factors for k in HF_FACTORS)]
        if hf_trades:
            lines += [
                "  ── 【快层·进场时机因子】盈利能力分析（LOBTimingEngine，逐tick，阈值触发下单，无权重）────",
                "  * 做空时 ofi/lob_z1/timing 取反，fast_boost/lob_pc1 方向独立",
                f"  {'因子':<22} {'IC(预测力)':>12} {'赢家均值':>10} {'输家均值':>10} {'差值':>8}",
            ]
            HF_FLIP = {"hf_microprice", "hf_ofi", "hf_lob_z1", "hf_lob_z2", "hf_timing_score"}

            def _hf_dir(t: "TradeRecord", fname: str, v: float) -> float:
                return (-v if t.side == "SHORT" else v) if fname in HF_FLIP else v

            rets = [t.ret_lev_pct for t in hf_trades]
            for fname in HF_FACTORS:
                vals = [_hf_dir(t, fname, t.factors.get(fname, 0.0)) for t in hf_trades]
                n_f  = len(vals)
                if n_f > 1:
                    mv   = sum(vals) / n_f
                    mr   = sum(rets) / n_f
                    cov  = sum((v - mv) * (r - mr) for v, r in zip(vals, rets))
                    sv   = (sum((v - mv) ** 2 for v in vals) / n_f) ** 0.5
                    sr   = (sum((r - mr) ** 2 for r in rets)   / n_f) ** 0.5
                    ic   = (cov / n_f / (sv * sr)) if sv > 1e-8 and sr > 1e-8 else 0.0
                else:
                    ic = 0.0
                win_v  = [_hf_dir(t, fname, t.factors.get(fname, 0.0)) for t in hf_trades if t.pnl_usdt > 0]
                loss_v = [_hf_dir(t, fname, t.factors.get(fname, 0.0)) for t in hf_trades if t.pnl_usdt <= 0]
                aw = sum(win_v)  / len(win_v)  if win_v  else 0.0
                al = sum(loss_v) / len(loss_v) if loss_v else 0.0
                lines.append(
                    f"  {fname:<22} {ic:>+11.3f}  {aw:>+9.3f}  {al:>+9.3f}  {aw-al:>+7.3f}"
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

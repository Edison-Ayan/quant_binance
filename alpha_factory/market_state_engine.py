"""
Market State Engine - 全市场环境感知

职责（系统"上帝视角"）：
    判断当前市场是否适合让这套截面 alpha 策略运转，
    以及如果运转，应该以什么强度运转。

输出五个维度：

    1. regime          : TRENDING / MEAN_REVERTING / VOLATILE / QUIET
       驱动因素：BTC 5m 趋势方向 + 横截面收益相关性 + 波动幅度

    2. dispersion      : float [0, ∞)
       横截面收益分散度（全市场 ret_1m 的标准差）
       越高 = 个股分化越大 = 截面策略 alpha 机会越多

    3. tradability     : float [0, 1]
       当前市场可交易健康度（spread 宽度 + 盘口深度）
       越低 = 手续费/滑点越高 = 越应该减少交易

    4. crowding_score  : float
       全市场资金费率均值的 Z-score（高正值 = 多头拥挤 = 做多谨慎）

    5. is_tradeable    : bool
       综合标志：tradability >= min_tradability 且 dispersion >= min_dispersion
       且非 VOLATILE regime 下已触发冲击保护

同时提供：
    regime_mult        : float [0.5, 1.0]  建议乘在 unified_alpha 上的 regime 折扣
    long_bias          : float [-1, 1]     当前市场做多/做空倾向（来自 BTC 方向）
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class MarketRegime(Enum):
    TRENDING      = "TRENDING"       # 明确方向趋势（截面 alpha 有效，跟方向）
    MEAN_REVERTING = "MEAN_REVERTING" # 均值回归市（截面 alpha 有效，逆势更好）
    VOLATILE      = "VOLATILE"       # 高波动混乱市（冲击风险高，降权重）
    QUIET         = "QUIET"          # 低波动无聊市（spread 成本侵蚀大，减少交易）


# ── 超参数 ────────────────────────────────────────────────────────────────────

# 最小参与品种数（用于横截面统计）
MIN_SYMBOLS = 10

# BTC 5m 涨跌幅判断趋势强度的阈值
TREND_STRONG_THRESH  = 0.015   # >1.5% → 强趋势
TREND_WEAK_THRESH    = 0.005   # 0.5~1.5% → 弱趋势
VOLATILE_VOL_THRESH  = 0.015   # 5m 横截面 std > 1.5% → 高波动（原 0.8% 对加密货币过于灵敏）
QUIET_ACTIVITY_THRESH = 0.1    # 全市场 volume_zscore 均值 < 0.1 → 市场静止

# 可交易性评分阈值
MIN_TRADABILITY    = 0.20      # 降低至 0.20（原 0.30），tradability 改为软权重，不再作为硬过滤
MAX_SPREAD_BPS     = 50.0      # 超过此 spread → 流动性不足
MIN_DEPTH_RATIO    = 0.3       # 盘口深度低于正常 30% → 深度不足

# 分散度阈值
MIN_DISPERSION     = 0.001     # 降低至 0.001（原 0.002），避免低波动期过度过滤

# 拥挤度
CROWDING_WINDOW    = 60        # 资金费率滚动窗口（轮）
CROWDING_Z_THRESH  = 2.0       # 拥挤度 Z-score 超过此值 → 警告

# regime_mult 映射（regime → 对 alpha 的置信度乘子）
REGIME_MULT = {
    MarketRegime.TRENDING:       1.00,
    MarketRegime.MEAN_REVERTING: 0.90,
    MarketRegime.VOLATILE:       0.60,
    MarketRegime.QUIET:          0.70,
}


@dataclass
class MarketState:
    """完整的市场状态快照（每次 update 后返回）"""
    regime:         MarketRegime = MarketRegime.QUIET
    dispersion:     float        = 0.0   # 横截面 ret_1m std
    tradability:    float        = 1.0   # [0, 1]
    crowding_score: float        = 0.0   # 资金费率 Z-score（正=多头拥挤）
    is_tradeable:   bool         = False
    regime_mult:    float        = 0.70  # 建议 alpha 折扣乘子
    long_bias:      float        = 0.0   # BTC 方向偏置 (-1 空/+1 多)
    avg_spread_bps: float        = 0.0
    avg_vol_zscore: float        = 0.0
    symbol_count:   int          = 0
    timestamp:      float        = 0.0


class MarketStateEngine:
    """
    全市场状态估计引擎。

    接口：
        update(features: dict) → MarketState
            features = {symbol: SymbolFeatures, ...}（来自 FeatureEngine.get_all_features()）

        get_state() → MarketState
            返回最近一次 update 的结果

    典型调用时机：每次慢层 rank 前调用一次（每 60s）
    """

    def __init__(self, btc_symbol: str = "BTCUSDT"):
        self._btc_symbol = btc_symbol
        self._current_state = MarketState()

        # 资金费率滚动窗口（用于计算拥挤度 Z-score）
        self._funding_hist: deque = deque(maxlen=CROWDING_WINDOW)
        self._funding_sum:    float = 0.0
        self._funding_sum_sq: float = 0.0

    # ─── 公开接口 ─────────────────────────────────────────────────────────────

    def update(self, features: dict) -> MarketState:
        """
        用当前全市场特征快照计算市场状态。

        features: {symbol: SymbolFeatures}（data_count >= 20 的才进入）
        """
        if len(features) < MIN_SYMBOLS:
            # 数据不足，保持当前状态但设 is_tradeable=False
            self._current_state.is_tradeable = False
            self._current_state.symbol_count = len(features)
            return self._current_state

        syms   = list(features.keys())
        n      = len(syms)

        # ── 1. 基础统计 ────────────────────────────────────────────────────
        ret1m_vals  = [features[s].ret_1m        for s in syms]
        ret5m_vals  = [features[s].ret_5m        for s in syms]
        spread_vals = [features[s].spread_bps    for s in syms if features[s].spread_bps > 0]
        vol_vals    = [abs(features[s].volume_zscore) for s in syms]
        fund_vals   = [features[s].funding_rate  for s in syms]

        avg_ret1m   = _mean(ret1m_vals)
        avg_ret5m   = _mean(ret5m_vals)
        std_ret1m   = _std(ret1m_vals)    # 横截面分散度
        avg_spread  = _mean(spread_vals) if spread_vals else 0.0
        avg_vol     = _mean(vol_vals)
        avg_funding = _mean(fund_vals)

        # ── 2. BTC 方向偏置 ────────────────────────────────────────────────
        btc_feat = features.get(self._btc_symbol)
        btc_ret5m = btc_feat.ret_5m if btc_feat else avg_ret5m
        long_bias = math.tanh(btc_ret5m / 0.01)   # [-1, 1] 软化

        # ── 3. 横截面分散度 ────────────────────────────────────────────────
        dispersion = std_ret1m  # 值越大 = 品种分化越大

        # ── 4. 市场制度判断 ────────────────────────────────────────────────
        abs_btc5m = abs(btc_ret5m)
        abs_5m_std = _std(ret5m_vals)

        if abs_5m_std > VOLATILE_VOL_THRESH:
            regime = MarketRegime.VOLATILE
        elif abs_btc5m >= TREND_STRONG_THRESH:
            regime = MarketRegime.TRENDING
        elif abs_btc5m >= TREND_WEAK_THRESH:
            # 弱趋势下看截面相关性：
            # 如果所有币和 BTC 方向一致 → 仍然是趋势；
            # 如果分化很大 → 均值回归
            same_dir = sum(1 for r in ret1m_vals if (r > 0) == (btc_ret5m > 0))
            if same_dir / n > 0.65:
                regime = MarketRegime.TRENDING
            else:
                regime = MarketRegime.MEAN_REVERTING
        elif avg_vol < QUIET_ACTIVITY_THRESH:
            regime = MarketRegime.QUIET
        else:
            regime = MarketRegime.MEAN_REVERTING

        # ── 5. 可交易性评分 ────────────────────────────────────────────────
        # spread 健康度：spread 越小越好，超过 MAX_SPREAD_BPS 则为 0
        spread_score = max(0.0, 1.0 - avg_spread / MAX_SPREAD_BPS)

        # 市场活跃度：volume_zscore 绝对均值越高越好
        activity_score = min(1.0, avg_vol / 1.0)

        # tradability = spread 健康度 × 活跃度，加权
        tradability = 0.6 * spread_score + 0.4 * activity_score
        tradability = max(0.0, min(1.0, tradability))

        # ── 6. 拥挤度 Z-score ──────────────────────────────────────────────
        crowding_score = self._update_funding_zscore(avg_funding)

        # ── 7. 综合可交易标志 ──────────────────────────────────────────────
        # tradability 不再作为硬过滤（改为 alpha_fusion 中的软权重乘子）
        # MEAN_REVERTING 无论 dispersion 多低都允许交易（均值回归策略本身适合此制度）
        if regime == MarketRegime.MEAN_REVERTING:
            is_tradeable = True
        elif regime == MarketRegime.VOLATILE:
            # VOLATILE：只有 tradability 很高时才允许小量操作
            is_tradeable = tradability >= 0.5
        else:
            # TRENDING / QUIET：要求基本分散度
            is_tradeable = dispersion >= MIN_DISPERSION

        state = MarketState(
            regime         = regime,
            dispersion     = round(dispersion, 6),
            tradability    = round(tradability, 4),
            crowding_score = round(crowding_score, 4),
            is_tradeable   = is_tradeable,
            regime_mult    = REGIME_MULT[regime],
            long_bias      = round(long_bias, 4),
            avg_spread_bps = round(avg_spread, 2),
            avg_vol_zscore = round(avg_vol, 4),
            symbol_count   = n,
            timestamp      = time.time(),
        )
        self._current_state = state
        return state

    def get_state(self) -> MarketState:
        return self._current_state

    def get_status(self) -> dict:
        s = self._current_state
        return {
            "regime":        s.regime.value,
            "dispersion":    s.dispersion,
            "tradability":   s.tradability,
            "crowding_z":    s.crowding_score,
            "is_tradeable":  s.is_tradeable,
            "regime_mult":   s.regime_mult,
            "long_bias":     s.long_bias,
            "avg_spread":    s.avg_spread_bps,
            "symbols":       s.symbol_count,
        }

    # ─── 内部方法 ─────────────────────────────────────────────────────────────

    def _update_funding_zscore(self, avg_funding: float) -> float:
        """
        维护资金费率滚动窗口，返回当前均值的 Z-score。
        正值 = 平均资金费率高于历史（多头拥挤）
        负值 = 空头拥挤
        """
        if len(self._funding_hist) == CROWDING_WINDOW:
            old = self._funding_hist[0]
            self._funding_sum    -= old
            self._funding_sum_sq -= old * old

        self._funding_hist.append(avg_funding)
        self._funding_sum    += avg_funding
        self._funding_sum_sq += avg_funding * avg_funding

        n = len(self._funding_hist)
        if n < 5:
            return 0.0

        mu  = self._funding_sum / n
        var = max(self._funding_sum_sq / n - mu * mu, 0.0)
        sig = var ** 0.5
        return (avg_funding - mu) / sig if sig > 1e-12 else 0.0


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    n  = len(vals)
    mu = sum(vals) / n
    return math.sqrt(max(sum((v - mu) ** 2 for v in vals) / n, 0.0))

"""
Alpha Fusion Engine - 统一 Alpha 表示（最终稳妥版）

设计目标：
    在不大改接口和字段的前提下，
    让 unified_alpha 更稳定、更符合中频交易系统的逻辑。

核心思想：
    把"慢层排名分"和"快层微结构信号"融合成一个统一的连续 alpha score，
    再叠加市场状态折扣和成本覆盖判断，输出真正可以驱动仓位大小的信号。

融合公式（保守版）：
    unified_alpha = slow_score
                  × regime_mult
                  × tradability_mult
                  × (1 + fast_boost)
                  × crowding_discount

与原版相比的主要修正：
    1. fast_boost / fast_drag 强度下降，避免 timing 过度改写 slow alpha
    2. 只有 |slow_score| 足够强时，fast_boost 才参与
    3. tradability floor 提高，避免 alpha 被乘得过小
    4. crowding 改成更稳的对称折扣，不再只惩罚做多
"""

import math
from dataclasses import dataclass
from typing import Dict

from .market_state_engine import MarketState


# ── 融合参数（保守版）──────────────────────────────────────────────────────────

# 只有 slow_score 足够强时，才允许 fast_boost 参与
MIN_SLOW_FOR_FAST = 0.50

# 快层加成：timing_score 与方向对齐时的最大提升（调低）
FAST_BOOST_MAX    =  0.30
FAST_BOOST_SCALE  =  0.15

# 快层减成：方向不对齐时的最大折扣（调低）
FAST_DRAG_MAX     = -0.25
FAST_DRAG_SCALE   =  0.10

# 拥挤度惩罚：改成更稳的对称折扣
CROWDING_PENALTY_THRESH = 1.5
CROWDING_PENALTY_SCALE  = 0.10
CROWDING_DISC_FLOOR     = 0.70

# unified_alpha 的 soft-clip 上下限（避免极端值破坏仓位计算）
UNIFIED_CLIP = 3.0

# ── 防"过度压缩"保护 ──────────────────────────────────────────────────────────

# tradability_mult 下限提高：避免流动性因子把 alpha 压得过小
TRADABILITY_MULT_FLOOR = 0.70


@dataclass
class FusedAlpha:
    """单个品种的融合 alpha 结果"""
    symbol:           str
    unified:          float = 0.0   # 最终融合 alpha
    slow_score:       float = 0.0   # 慢层 Z-score
    fast_boost_val:   float = 0.0   # 快层实际贡献（乘法调节项）
    regime_mult:      float = 1.0
    tradability_mult: float = 1.0
    crowding_disc:    float = 1.0
    is_long_candidate:  bool = False
    is_short_candidate: bool = False


class AlphaFusionEngine:
    """
    Alpha 融合引擎。

    接口保持不变：
        fuse(scores, features, timing_engine, market_state) → Dict[str, FusedAlpha]

    参数：
        scores         : Dict[symbol, float] — ScoringEngine 输出的截面分数
        features       : Dict[symbol, SymbolFeatures] — FeatureEngine 快照
        timing_engine  : LOBTimingEngine 实例（用于读取 timing_score）
        market_state   : MarketState 实例（来自 MarketStateEngine）

    返回：
        {symbol: FusedAlpha}
    """

    def __init__(self, entry_threshold: float = 0.30):
        self.entry_threshold = entry_threshold

    def fuse(
        self,
        scores: Dict[str, float],
        features: dict,
        timing_engine,
        market_state: MarketState,
    ) -> Dict[str, "FusedAlpha"]:
        """
        融合所有品种的 alpha 信号，返回 {symbol: FusedAlpha}。
        """
        result: Dict[str, FusedAlpha] = {}

        regime_mult      = market_state.regime_mult
        tradability_mult = market_state.tradability
        crowding_z       = market_state.crowding_score

        # tradability 软权重：加下限，不允许被乘到过小
        tradability_applied = max(tradability_mult, TRADABILITY_MULT_FLOOR)

        for sym, slow_score in scores.items():
            feat = features.get(sym)
            if feat is None:
                continue

            # ── 快层加成（保守版）───────────────────────────────────────────
            timing_score = timing_engine.get_timing_score(sym, feat) if timing_engine else 0.0
            fast_boost = self._calc_fast_boost(slow_score, timing_score)

            # ── 拥挤度折扣（改为对称、更稳）────────────────────────────────
            crowding_disc = self._calc_crowding_discount(crowding_z)

            # ── 融合（接口不变，仍输出 unified）────────────────────────────
            unified = (
                slow_score
                * regime_mult
                * tradability_applied
                * (1.0 + fast_boost)
                * crowding_disc
            )

            unified = _soft_clip(unified, UNIFIED_CLIP)

            result[sym] = FusedAlpha(
                symbol             = sym,
                unified            = round(unified, 5),
                slow_score         = round(slow_score, 5),
                fast_boost_val     = round(fast_boost, 5),
                regime_mult        = round(regime_mult, 4),
                tradability_mult   = round(tradability_applied, 4),
                crowding_disc      = round(crowding_disc, 4),
                is_long_candidate  = unified >= self.entry_threshold,
                is_short_candidate = unified <= -self.entry_threshold,
            )

        return result

    def get_top_candidates(
        self,
        fused: Dict[str, FusedAlpha],
        top_n: int,
        side: str,
    ) -> list:
        """
        返回 Top-N 候选列表（按 unified alpha 强度排序）。
        side: "long" 或 "short"
        """
        if side == "long":
            candidates = [(s, fa) for s, fa in fused.items() if fa.is_long_candidate]
            return sorted(candidates, key=lambda x: x[1].unified, reverse=True)[:top_n]
        else:
            candidates = [(s, fa) for s, fa in fused.items() if fa.is_short_candidate]
            return sorted(candidates, key=lambda x: x[1].unified)[:top_n]

    # ─── 内部计算 ─────────────────────────────────────────────────────────────

    def _calc_fast_boost(self, slow_score: float, timing_score: float) -> float:
        """
        保守版 fast_boost：
            1. 只有 |slow_score| >= MIN_SLOW_FOR_FAST 时才启用
            2. 对齐时小幅增强
            3. 反向时小幅减弱
            4. 不允许 timing 过度改写 slow alpha
        """
        if abs(slow_score) < MIN_SLOW_FOR_FAST:
            return 0.0

        # 做多候选：timing 正向 = 加成，负向 = 减成
        if slow_score >= 0:
            if timing_score > 0:
                return min(FAST_BOOST_SCALE * timing_score, FAST_BOOST_MAX)
            return max(FAST_DRAG_SCALE * timing_score, FAST_DRAG_MAX)

        # 做空候选：timing 负向 = 加成，正向 = 减成
        if timing_score < 0:
            return min(FAST_BOOST_SCALE * abs(timing_score), FAST_BOOST_MAX)

        drag = -FAST_DRAG_SCALE * timing_score
        return max(drag, FAST_DRAG_MAX)

    def _calc_crowding_discount(self, crowding_z: float) -> float:
        """
        更稳的对称拥挤度折扣：
            |crowding_z| 超过阈值后开始降低 confidence
            但保留底线，不让其无限压缩
        """
        abs_crowding = abs(crowding_z)
        if abs_crowding <= CROWDING_PENALTY_THRESH:
            return 1.0

        excess = abs_crowding - CROWDING_PENALTY_THRESH
        disc = 1.0 - CROWDING_PENALTY_SCALE * excess
        return max(CROWDING_DISC_FLOOR, disc)


def _soft_clip(x: float, scale: float) -> float:
    return math.tanh(x / scale) * scale
"""
Alpha Lifecycle State Machine

范式升级：
    旧版：if price >= entry * 1.015 → take_profit（价格驱动）
    新版：持续追踪每个持仓的 alpha 逻辑是否仍然成立（逻辑驱动）

Alpha 状态机：
    BUILD     → alpha 刚形成，强度尚未完全确认，试探性建仓期
    EXPANSION → alpha 方向 + 强度双重强化（slow/fast 共振），最佳持有期
    DECAY     → alpha 方向未反转，但边际优势在下降，开始减仓/锁利
    REVERSAL  → alpha 方向翻转，应快速退出

转换条件（以 LONG 为例，SHORT 取反）：
    BUILD → EXPANSION  : unified_score > expand_thresh 且 velocity > 0
    BUILD → DECAY      : unified_score 长时间未进入扩张，或持仓超 build_max_s
    EXPANSION → DECAY  : velocity < 0 连续若干轮确认
    DECAY → EXPANSION  : velocity 重新转正且 score 回升
    DECAY → REVERSAL   : alpha 明确反向，或持续恶化且强度显著衰减
    * → REVERSAL       : unified_score 方向明确翻转

持仓管理建议（外部调用方参考 action_hint）：
    BUILD     → hold，仅试探性仓位，不加仓
    EXPANSION → hold / 可小量加仓（由 PortfolioConstructor 决定）
    DECAY     → 开始减仓，收紧 trailing stop
    REVERSAL  → 立即出场，不等价格止损
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class AlphaState(Enum):
    BUILD = "BUILD"
    EXPANSION = "EXPANSION"
    DECAY = "DECAY"
    REVERSAL = "REVERSAL"


# ── 状态机超参数 ───────────────────────────────────────────────────────────────

# BUILD → EXPANSION：score 超过此值且 velocity > 0
EXPAND_THRESH = 0.60

# 任意状态下：只有当 alpha 明确翻转到这个阈值以下，才算真正 REVERSAL
# 原版 -0.20 太浅，容易把噪声误判成翻转
REVERSAL_THRESH = -0.35

# 速度 EWMA 衰减系数（低 α = 更平滑，更保守）
VEL_EMA_ALPHA = 0.25

# 连续 N 轮 velocity < 0 才确认 EXPANSION → DECAY
# 原版是 1，太敏感，容易一轮噪声就从 EXPANSION 打到 DECAY
VEL_DECAY_ROUNDS = 2

# 最长 BUILD 持续时间（秒）：超过后若无 EXPANSION 迹象则降为 DECAY
BUILD_MAX_SEC = 300.0

# REVERSAL 触发后还要重新等到此阈值才能恢复（防止频繁抖动）
REVERSAL_RECOVERY_THRESH = 0.40

# DECAY 状态下，若 alpha 相对峰值衰减到该比例以下，认为已经明显弱化
DECAY_DEEP_RATIO = 0.60

# DECAY 状态下，alpha 已明显变弱后，还需要连续多少轮负速度，才升级为 REVERSAL
DECAY_REVERSAL_NEG_VEL_ROUNDS = 2

# DECAY 状态下，若 aligned_score 低于该值，说明 alpha 本体已偏弱
DECAY_WEAK_SCORE = 0.30


@dataclass
class PositionLifecycle:
    """单个持仓的 alpha 生命周期状态"""
    symbol: str
    side: str  # "LONG" / "SHORT"
    state: AlphaState = AlphaState.BUILD

    # Score 追踪
    score_entry: float = 0.0   # 开仓时的 unified alpha score
    score_peak: float = 0.0    # 持仓期间最佳 aligned score
    score_current: float = 0.0 # 当前 unified alpha score
    score_prev: float = 0.0    # 上一轮 score（用于速度计算）

    # 速度（EWMA 平滑的 score 变化率）
    velocity: float = 0.0

    # 计数器
    neg_vel_count: int = 0
    pos_vel_count: int = 0

    # 时间戳
    open_time: float = field(default_factory=time.time)
    state_entered_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def _aligned_score(self) -> float:
        """
        方向对齐的 score：
        LONG 持仓期望 unified_score 为正
        SHORT 持仓期望 unified_score 为负，这里取反转成正数
        """
        return self.score_current if self.side == "LONG" else -self.score_current

    def _aligned_vel(self) -> float:
        """
        方向对齐的速度：
        LONG 希望 velocity > 0
        SHORT 希望 velocity < 0，这里取反转成“越大越好”
        """
        return self.velocity if self.side == "LONG" else -self.velocity


class AlphaLifecycleTracker:
    """
    管理所有持仓的 alpha 生命周期状态。

    外部调用：
        open_position(symbol, side, score)   → 注册新持仓
        update(symbol, side, unified_score)  → 每次排名后更新
        get_state(symbol, side)              → 查询当前状态
        close_position(symbol, side)         → 持仓平仓后清理
        get_action_hint(symbol, side)        → 返回建议动作
        get_all_status()                     → 所有持仓状态摘要
    """

    def __init__(self):
        # key: (symbol, side)
        self._positions: Dict[tuple, PositionLifecycle] = {}

    # ─── 公开接口 ─────────────────────────────────────────────────────────

    def open_position(self, symbol: str, side: str, score: float):
        now = time.time()
        aligned_score = score if side == "LONG" else -score

        lc = PositionLifecycle(
            symbol=symbol,
            side=side,
            state=AlphaState.BUILD,
            score_entry=score,
            score_peak=max(aligned_score, 0.0),
            score_current=score,
            score_prev=score,
            open_time=now,
            state_entered_at=now,
            last_updated_at=now,
        )
        self._positions[(symbol, side)] = lc

    def close_position(self, symbol: str, side: str):
        self._positions.pop((symbol, side), None)

    def update(self, symbol: str, side: str, unified_score: float) -> AlphaState:
        """
        用最新的 unified_alpha score 更新生命周期状态，返回新状态。
        建议每次 slow layer 排名完成后调用。
        """
        key = (symbol, side)
        lc = self._positions.get(key)
        if lc is None:
            return AlphaState.BUILD

        now = time.time()

        # ── 更新 score / velocity ────────────────────────────────────────
        raw_vel = unified_score - lc.score_current
        lc.velocity = VEL_EMA_ALPHA * raw_vel + (1.0 - VEL_EMA_ALPHA) * lc.velocity
        lc.score_prev = lc.score_current
        lc.score_current = unified_score
        lc.last_updated_at = now

        aligned_score = lc._aligned_score()
        aligned_vel = lc._aligned_vel()

        # 更新峰值（只记录方向一致时的正向峰值）
        if aligned_score > lc.score_peak:
            lc.score_peak = aligned_score

        # ── 更新速度计数器 ──────────────────────────────────────────────
        if aligned_vel < 0:
            lc.neg_vel_count += 1
            lc.pos_vel_count = 0
        elif aligned_vel > 0:
            lc.pos_vel_count += 1
            lc.neg_vel_count = 0
        # = 0 时不变，避免无意义抖动重置

        old_state = lc.state
        new_state = self._transition(lc, now)

        if new_state != old_state:
            lc.state = new_state
            lc.state_entered_at = now

            # 状态切换后，计数器清一下，避免旧状态残留影响新状态
            lc.neg_vel_count = 0
            lc.pos_vel_count = 0

        return lc.state

    def get_state(self, symbol: str, side: str) -> Optional[AlphaState]:
        lc = self._positions.get((symbol, side))
        return lc.state if lc else None

    def get_lifecycle(self, symbol: str, side: str) -> Optional[PositionLifecycle]:
        return self._positions.get((symbol, side))

    def get_action_hint(self, symbol: str, side: str) -> str:
        """
        返回建议动作字符串：
            "hold"    → 正常持有
            "add"     → 可考虑加仓（EXPANSION 阶段）
            "reduce"  → 建议减仓（DECAY 阶段）
            "exit"    → 立即出场（REVERSAL）
        """
        state = self.get_state(symbol, side)
        if state is None:
            return "hold"

        return {
            AlphaState.BUILD: "hold",
            AlphaState.EXPANSION: "add",
            AlphaState.DECAY: "reduce",
            AlphaState.REVERSAL: "exit",
        }[state]

    def get_all_status(self) -> dict:
        now = time.time()
        result = {}

        for (sym, side), lc in self._positions.items():
            result[f"{sym}_{side}"] = {
                "state": lc.state.value,
                "score": round(lc.score_current, 3),
                "aligned_score": round(lc._aligned_score(), 3),
                "velocity": round(lc.velocity, 4),
                "aligned_velocity": round(lc._aligned_vel(), 4),
                "peak": round(lc.score_peak, 3),
                "hold_s": round(now - lc.open_time, 0),
                "state_age_s": round(now - lc.state_entered_at, 0),
                "action": self.get_action_hint(sym, side),
            }

        return result

    # ─── 状态转换逻辑 ─────────────────────────────────────────────────────

    def _transition(self, lc: PositionLifecycle, now: float) -> AlphaState:
        aligned_score = lc._aligned_score()
        aligned_vel = lc._aligned_vel()
        age_in_state = now - lc.state_entered_at

        # ── 任意状态下：只有明确反向才算 REVERSAL ────────────────────────
        if aligned_score < REVERSAL_THRESH:
            return AlphaState.REVERSAL

        # ── BUILD ────────────────────────────────────────────────────────
        if lc.state == AlphaState.BUILD:
            # BUILD → EXPANSION：score 足够强且速度向上
            if aligned_score >= EXPAND_THRESH and aligned_vel > 0:
                return AlphaState.EXPANSION

            # BUILD → DECAY：停留过久仍未进入扩张
            if age_in_state > BUILD_MAX_SEC:
                return AlphaState.DECAY

            return AlphaState.BUILD

        # ── EXPANSION ───────────────────────────────────────────────────
        elif lc.state == AlphaState.EXPANSION:
            # EXPANSION → DECAY：连续若干轮速度为负，确认不是单点噪声
            if lc.neg_vel_count >= VEL_DECAY_ROUNDS:
                return AlphaState.DECAY

            return AlphaState.EXPANSION

        # ── DECAY ───────────────────────────────────────────────────────
        elif lc.state == AlphaState.DECAY:
            # DECAY → EXPANSION：重新恢复强 alpha + 正速度
            if (
                lc.pos_vel_count >= VEL_DECAY_ROUNDS
                and aligned_score >= EXPAND_THRESH
            ):
                return AlphaState.EXPANSION

            # 关键修正：
            # “alpha 从峰值回撤” 不等于 “方向反转”
            # 所以这里不直接 REVERSAL，而是要求：
            # 1) alpha 已明显衰减
            # 2) 且还在持续恶化（连续负速度）
            peak_ratio = 1.0
            if lc.score_peak > 1e-8:
                peak_ratio = aligned_score / lc.score_peak

            decay_is_deep = peak_ratio < DECAY_DEEP_RATIO
            alpha_is_weak = aligned_score < DECAY_WEAK_SCORE
            still_worsening = lc.neg_vel_count >= DECAY_REVERSAL_NEG_VEL_ROUNDS

            if (decay_is_deep and still_worsening) or (alpha_is_weak and still_worsening):
                return AlphaState.REVERSAL

            return AlphaState.DECAY

        # ── REVERSAL ────────────────────────────────────────────────────
        elif lc.state == AlphaState.REVERSAL:
            # 冷却后允许恢复到 BUILD，避免抖动反复横跳
            if aligned_score >= REVERSAL_RECOVERY_THRESH and age_in_state > 60.0:
                return AlphaState.BUILD

            return AlphaState.REVERSAL

        return lc.state
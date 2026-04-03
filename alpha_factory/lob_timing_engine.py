"""
LOB Timing Engine - 基于微观结构的实时进场时机判断（强化版）

正确定位：
    不预测方向（方向由横截面 Alpha 的候选池决定）
    只判断：候选池内的品种 "现在适不适合进"

升级点：
    1. 连续确认：不是一次过阈值就进，而是需连续 N 次满足
    2. 方向一致性：做多要求 microprice / OFI 不反向；做空反之
    3. 更严格阈值：提高 entry / exit threshold，减少噪声触发
    4. 保留 soft_clip + rolling zscore，避免极端值破坏稳定性

核心公式：
    timing_score = w_micro  × microprice_delta_zscore
                 + w_ofi    × ofi_soft_clip
                 + w_pc1    × lob_pc1_soft_clip
                 + w_z1     × lob_z1_soft_clip
                 + w_z2     × lob_z2_soft_clip
                 + w_z3     × lob_z3_soft_clip

触发规则（强化版）：
    开多：
        timing_score > entry_threshold
        且 ofi > 0
        且 microprice_zscore > 0
        且连续满足 confirm_ticks 次

    开空：
        timing_score < -entry_threshold
        且 ofi < 0
        且 microprice_zscore < 0
        且连续满足 confirm_ticks 次

    alpha_flip 出多：timing_score < -exit_threshold
    alpha_flip 出空：timing_score > +exit_threshold
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from data_layer.logger import logger


# ── 默认权重（合计 1.0）───────────────────────────────────────────────────────
W_MICROPRICE = 0.19
W_OFI        = 0.21
W_LOB_PC1    = 0.14
W_LOB_Z1     = 0.33
W_LOB_Z2     = 0.09
W_LOB_Z3     = 0.04

# ── 进场/出场阈值（强化版）─────────────────────────────────────────────────────
# entry > exit：在 [-exit, +entry] 之间形成死区，防止反复触发
ENTRY_THRESHOLD: float = 0.45
EXIT_THRESHOLD:  float = 0.25

# ── microprice_delta 滚动归一化窗口 ───────────────────────────────────────────
NORM_WINDOW: int = 200   # 对应约20-30秒的 bookTicker 更新

# ── 进场冷却（防止同一信号在同一品种多次触发开仓）─────────────────────────────
ENTRY_COOLDOWN_S: float = 5.0

# ── 连续确认（新增）───────────────────────────────────────────────────────────
ENTRY_CONFIRM_TICKS: int = 2


@dataclass
class _TimingState:
    """单个品种的微观结构状态（内部使用）"""
    delta_buf:          deque = field(default_factory=lambda: deque(maxlen=NORM_WINDOW))
    delta_sum:          float = 0.0
    delta_sum_sq:       float = 0.0
    last_delta:         float = 0.0   # 最新原始 microprice_delta（入队后更新）
    last_entry_ts:      float = 0.0   # 最近一次触发进场的时间戳

    # 新增：连续确认计数
    long_confirm_count: int = 0
    short_confirm_count:int = 0


class LOBTimingEngine:
    """
    实时进场时机判断引擎（强化版）。

    数据流：
        on_book_ticker()   → bookTicker 每次推送时调用，更新 microprice_delta
        get_timing_score() → 读取 microprice + feat.ofi + feat.lob，返回连续得分
        should_enter_*()   → 加入连续确认 + 方向一致性后决定是否进场
        should_exit_*()    → timing_score 反向超阈值时返回 True
        record_entry()     → 开仓后调用，启动冷却期并清空确认计数
    """

    def __init__(
        self,
        w_microprice:   float = W_MICROPRICE,
        w_ofi:          float = W_OFI,
        w_lob_pc1:      float = W_LOB_PC1,
        w_lob_z1:       float = W_LOB_Z1,
        w_lob_z2:       float = W_LOB_Z2,
        w_lob_z3:       float = W_LOB_Z3,
        entry_threshold:float = ENTRY_THRESHOLD,
        exit_threshold: float = EXIT_THRESHOLD,
        confirm_ticks:  int   = ENTRY_CONFIRM_TICKS,
    ):
        self.w_microprice    = w_microprice
        self.w_ofi           = w_ofi
        self.w_lob_pc1       = w_lob_pc1
        self.w_lob_z1        = w_lob_z1
        self.w_lob_z2        = w_lob_z2
        self.w_lob_z3        = w_lob_z3
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold
        self.confirm_ticks   = max(1, confirm_ticks)

        self._states: Dict[str, _TimingState] = {}

    # ─── 数据更新 ─────────────────────────────────────────────────────────────

    def on_book_ticker(
        self,
        symbol:  str,
        bid:     float,
        bid_qty: float,
        ask:     float,
        ask_qty: float,
    ):
        """
        每次 bookTicker 推送时调用，维护 microprice_delta 的滚动归一化状态。
        """
        total_qty = bid_qty + ask_qty
        if total_qty < 1e-12 or bid <= 0 or ask <= 0:
            return

        microprice = (bid * ask_qty + ask * bid_qty) / total_qty
        simple_mid = (bid + ask) * 0.5
        delta      = microprice - simple_mid

        st = self._get_state(symbol)

        # O(1) 滚动均值/方差更新
        if len(st.delta_buf) == NORM_WINDOW:
            old = st.delta_buf[0]
            st.delta_sum    -= old
            st.delta_sum_sq -= old * old

        st.delta_buf.append(delta)
        st.delta_sum    += delta
        st.delta_sum_sq += delta * delta
        st.last_delta    = delta

    # ─── 打分 ─────────────────────────────────────────────────────────────────

    def get_timing_score(self, symbol: str, features) -> float:
        """
        返回 timing_score（连续值，通常在 [-3, +3] 左右）。

        features 为 SymbolFeatures，提供：
            ofi / lob_pc1 / lob_z1 / lob_z2 / lob_z3
        """
        micro_norm = self._microprice_zscore(symbol)
        ofi        = getattr(features, "ofi",     0.0)
        lob_pc1    = getattr(features, "lob_pc1", 0.0)
        lob_z1     = getattr(features, "lob_z1",  0.0)
        lob_z2     = getattr(features, "lob_z2",  0.0)
        lob_z3     = getattr(features, "lob_z3",  0.0)

        return (
            self.w_microprice * micro_norm
            + self.w_ofi      * _soft_clip(ofi,     scale=1.0)
            + self.w_lob_pc1  * _soft_clip(lob_pc1, scale=1.0)
            + self.w_lob_z1   * _soft_clip(lob_z1,  scale=1.0)
            + self.w_lob_z2   * _soft_clip(lob_z2,  scale=1.0)
            + self.w_lob_z3   * _soft_clip(lob_z3,  scale=1.0)
        )

    # ─── 进场判断（强化版）────────────────────────────────────────────────────

    def should_enter_long(self, symbol: str, features) -> bool:
        """
        做多进场条件（强化版）：
            1. 不在冷却期
            2. timing_score > entry_threshold
            3. ofi > 0
            4. microprice_zscore > 0
            5. 连续满足 confirm_ticks 次
        """
        st = self._get_state(symbol)

        if self._cooling_down(symbol):
            st.long_confirm_count = 0
            return False

        score = self.get_timing_score(symbol, features)
        micro_norm = self._microprice_zscore(symbol)
        ofi = getattr(features, "ofi", 0.0)

        # 方向一致性：做多时必须至少 OFI / microprice 不反向
        if score > self.entry_threshold and ofi > 0 and micro_norm > 0:
            st.long_confirm_count += 1
        else:
            st.long_confirm_count = 0

        # 多空互斥确认计数
        st.short_confirm_count = 0

        return st.long_confirm_count >= self.confirm_ticks

    def should_enter_short(self, symbol: str, features) -> bool:
        """
        做空进场条件（强化版）：
            1. 不在冷却期
            2. timing_score < -entry_threshold
            3. ofi < 0
            4. microprice_zscore < 0
            5. 连续满足 confirm_ticks 次
        """
        st = self._get_state(symbol)

        if self._cooling_down(symbol):
            st.short_confirm_count = 0
            return False

        score = self.get_timing_score(symbol, features)
        micro_norm = self._microprice_zscore(symbol)
        ofi = getattr(features, "ofi", 0.0)

        # 方向一致性：做空时必须至少 OFI / microprice 不反向
        if score < -self.entry_threshold and ofi < 0 and micro_norm < 0:
            st.short_confirm_count += 1
        else:
            st.short_confirm_count = 0

        # 多空互斥确认计数
        st.long_confirm_count = 0

        return st.short_confirm_count >= self.confirm_ticks

    # ─── 出场判断 ─────────────────────────────────────────────────────────────

    def should_exit_long(self, symbol: str, features) -> bool:
        """timing 反向确认 → alpha_flip 出多"""
        return self.get_timing_score(symbol, features) < -self.exit_threshold

    def should_exit_short(self, symbol: str, features) -> bool:
        """timing 反向确认 → alpha_flip 出空"""
        return self.get_timing_score(symbol, features) > self.exit_threshold

    def record_entry(self, symbol: str):
        """
        开仓后调用：
            1. 启动冷却期，防止同一信号多次触发
            2. 清空确认计数
        """
        st = self._get_state(symbol)
        st.last_entry_ts = time.time()
        st.long_confirm_count = 0
        st.short_confirm_count = 0

    # ─── 调试输出 ─────────────────────────────────────────────────────────────

    def get_score_components(self, symbol: str, features) -> dict:
        """调试用：返回各分量的实际贡献值 + 确认状态"""
        st = self._get_state(symbol)

        micro_norm = self._microprice_zscore(symbol)
        ofi        = getattr(features, "ofi",     0.0)
        lob_pc1    = getattr(features, "lob_pc1", 0.0)
        lob_z1     = getattr(features, "lob_z1",  0.0)
        lob_z2     = getattr(features, "lob_z2",  0.0)
        lob_z3     = getattr(features, "lob_z3",  0.0)

        ofi_clip = _soft_clip(ofi, 1.0)
        pc1_clip = _soft_clip(lob_pc1, 1.0)
        z1_clip  = _soft_clip(lob_z1, 1.0)
        z2_clip  = _soft_clip(lob_z2, 1.0)
        z3_clip  = _soft_clip(lob_z3, 1.0)

        total = self.get_timing_score(symbol, features)

        return {
            "microprice_norm":    round(micro_norm, 4),
            "ofi_clip":           round(ofi_clip,   4),
            "lob_pc1_clip":       round(pc1_clip,   4),
            "lob_z1_clip":        round(z1_clip,    4),
            "lob_z2_clip":        round(z2_clip,    4),
            "lob_z3_clip":        round(z3_clip,    4),

            "microprice_contrib": round(self.w_microprice * micro_norm, 4),
            "ofi_contrib":        round(self.w_ofi        * ofi_clip,   4),
            "lob_pc1_contrib":    round(self.w_lob_pc1    * pc1_clip,   4),
            "lob_z1_contrib":     round(self.w_lob_z1     * z1_clip,    4),
            "lob_z2_contrib":     round(self.w_lob_z2     * z2_clip,    4),
            "lob_z3_contrib":     round(self.w_lob_z3     * z3_clip,    4),

            "total":              round(total, 4),

            "long_confirm_count":  st.long_confirm_count,
            "short_confirm_count": st.short_confirm_count,
            "confirm_ticks":       self.confirm_ticks,

            "long_alignment_ok":   bool(ofi > 0 and micro_norm > 0),
            "short_alignment_ok":  bool(ofi < 0 and micro_norm < 0),
        }

    # ─── 内部工具 ─────────────────────────────────────────────────────────────

    def _get_state(self, symbol: str) -> _TimingState:
        if symbol not in self._states:
            self._states[symbol] = _TimingState()
        return self._states[symbol]

    def _microprice_zscore(self, symbol: str) -> float:
        """
        microprice_delta 的滚动 z-score，使用 tanh 做软截断，
        输出大致落在 [-3, +3]。
        """
        st = self._states.get(symbol)
        if st is None or len(st.delta_buf) < 20:
            return 0.0

        n = len(st.delta_buf)
        mu = st.delta_sum / n
        var = max(st.delta_sum_sq / n - mu * mu, 0.0)
        sigma = var ** 0.5

        if sigma < 1e-14:
            return 0.0

        raw_z = (st.last_delta - mu) / sigma
        return _soft_clip(raw_z, scale=3.0)

    def _cooling_down(self, symbol: str) -> bool:
        st = self._states.get(symbol)
        return st is not None and (time.time() - st.last_entry_ts) < ENTRY_COOLDOWN_S


def _soft_clip(x: float, scale: float = 3.0) -> float:
    """
    tanh 软截断：
        保留方向和相对大小，消除极端值的破坏性影响
    """
    if scale <= 0:
        return x
    return math.tanh(x / scale) * scale
"""
LOB Timing Engine - 基于微观结构的实时进场时机判断

正确定位：
    不预测方向（方向由横截面 Alpha 的候选池决定）
    只判断：候选池内的品种 "现在适不适合进"

核心公式：
    timing_score = w_micro × microprice_delta_zscore
                 + w_ofi   × ofi_soft_clip
                 + w_z2    × lob_z2_soft_clip

分量含义：
    microprice_delta  = microprice - simple_mid
        microprice = (bid × ask_qty + ask × bid_qty) / (bid_qty + ask_qty)
        > 0 → ask 侧吃单大 → 买压  /  < 0 → bid 侧吃单大 → 卖压
        直觉：做市商在 ask 侧挂大量时，买方来吃单 → microprice 向 ask 偏移

    ofi_ema           来自 FeatureEngine（EWMA 订单流失衡）
    lob_z2            来自 LOBManifoldEngine（买卖不对称 PCA 坐标）

触发规则（无硬过滤，连续阈值）：
    开多：timing_score > entry_threshold
    开空：timing_score < -entry_threshold
    alpha_flip 出多：timing_score < -exit_threshold
    alpha_flip 出空：timing_score > +exit_threshold
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

from data_layer.logger import logger


# ── 默认权重 ───────────────────────────────────────────────────────────────────
W_MICROPRICE: float = 0.40
W_OFI:        float = 0.35
W_LOB_Z2:     float = 0.25

# ── 进场/出场阈值 ──────────────────────────────────────────────────────────────
# entry > exit：在 [-exit, +entry] 之间形成死区，防止反复触发
ENTRY_THRESHOLD: float = 0.40
EXIT_THRESHOLD:  float = 0.20

# ── microprice_delta 滚动归一化窗口 ───────────────────────────────────────────
NORM_WINDOW: int = 200   # 对应约20-30秒的 bookTicker 更新

# ── 进场冷却（防止同一信号在同一品种多次触发开仓）─────────────────────────────
ENTRY_COOLDOWN_S: float = 5.0


@dataclass
class _TimingState:
    """单个品种的微观结构状态（内部使用）"""
    delta_buf:     deque = field(default_factory=lambda: deque(maxlen=NORM_WINDOW))
    delta_sum:     float = 0.0
    delta_sum_sq:  float = 0.0
    last_delta:    float = 0.0   # 最新原始 microprice_delta（入队后更新）
    last_entry_ts: float = 0.0   # 最近一次触发进场的时间戳


class LOBTimingEngine:
    """
    实时进场时机判断引擎。

    数据流：
        on_book_ticker()  → bookTicker 每次推送时调用，更新 microprice_delta
        get_timing_score()→ 读取 microprice + feat.ofi + feat.lob_z2，返回连续得分
        should_enter_*()  → timing_score 超阈值时返回 True
        should_exit_*()   → timing_score 反向超阈值时返回 True
        record_entry()    → 开仓后调用，启动冷却期
    """

    def __init__(
        self,
        w_microprice:    float = W_MICROPRICE,
        w_ofi:           float = W_OFI,
        w_lob_z2:        float = W_LOB_Z2,
        entry_threshold: float = ENTRY_THRESHOLD,
        exit_threshold:  float = EXIT_THRESHOLD,
    ):
        self.w_microprice    = w_microprice
        self.w_ofi           = w_ofi
        self.w_lob_z2        = w_lob_z2
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold

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
        返回 timing_score（连续值，无截断，约在 [-3, +3] 之间）。

        features 为 SymbolFeatures，提供 ofi 和 lob_z2。
        """
        micro_norm = self._microprice_zscore(symbol)
        ofi        = getattr(features, "ofi",    0.0)
        lob_z2     = getattr(features, "lob_z2", 0.0)

        return (
            self.w_microprice * micro_norm
            + self.w_ofi      * _soft_clip(ofi,    scale=1.0)
            + self.w_lob_z2   * _soft_clip(lob_z2, scale=1.0)
        )

    def should_enter_long(self, symbol: str, features) -> bool:
        if self._cooling_down(symbol):
            return False
        return self.get_timing_score(symbol, features) > self.entry_threshold

    def should_enter_short(self, symbol: str, features) -> bool:
        if self._cooling_down(symbol):
            return False
        return self.get_timing_score(symbol, features) < -self.entry_threshold

    def should_exit_long(self, symbol: str, features) -> bool:
        """timing 反向确认 → alpha_flip 出多"""
        return self.get_timing_score(symbol, features) < -self.exit_threshold

    def should_exit_short(self, symbol: str, features) -> bool:
        """timing 反向确认 → alpha_flip 出空"""
        return self.get_timing_score(symbol, features) > self.exit_threshold

    def record_entry(self, symbol: str):
        """开仓后调用，启动冷却期防止同一信号多次触发"""
        self._get_state(symbol).last_entry_ts = time.time()

    def get_score_components(self, symbol: str, features) -> dict:
        """调试用：返回各分量的实际贡献值"""
        micro_norm = self._microprice_zscore(symbol)
        ofi        = getattr(features, "ofi",    0.0)
        lob_z2     = getattr(features, "lob_z2", 0.0)
        ofi_clip   = _soft_clip(ofi,    1.0)
        z2_clip    = _soft_clip(lob_z2, 1.0)
        return {
            "microprice_norm":  round(micro_norm, 4),
            "ofi_clip":         round(ofi_clip,   4),
            "lob_z2_clip":      round(z2_clip,    4),
            "microprice_contrib": round(self.w_microprice * micro_norm, 4),
            "ofi_contrib":        round(self.w_ofi * ofi_clip,          4),
            "z2_contrib":         round(self.w_lob_z2 * z2_clip,        4),
            "total":              round(self.get_timing_score(symbol, features), 4),
        }

    # ─── 内部工具 ─────────────────────────────────────────────────────────────

    def _get_state(self, symbol: str) -> _TimingState:
        if symbol not in self._states:
            self._states[symbol] = _TimingState()
        return self._states[symbol]

    def _microprice_zscore(self, symbol: str) -> float:
        """microprice_delta 的滚动 z-score，tanh 软截断至约 [-3, +3]"""
        st = self._states.get(symbol)
        if st is None or len(st.delta_buf) < 20:
            return 0.0
        n     = len(st.delta_buf)
        mu    = st.delta_sum / n
        var   = max(st.delta_sum_sq / n - mu * mu, 0.0)
        sigma = var ** 0.5
        if sigma < 1e-14:
            return 0.0
        raw_z = (st.last_delta - mu) / sigma
        return _soft_clip(raw_z, scale=3.0)

    def _cooling_down(self, symbol: str) -> bool:
        st = self._states.get(symbol)
        return st is not None and (time.time() - st.last_entry_ts) < ENTRY_COOLDOWN_S


def _soft_clip(x: float, scale: float = 3.0) -> float:
    """tanh 软截断：保留方向和相对大小，消除极端值的破坏性影响"""
    if scale <= 0:
        return x
    return math.tanh(x / scale) * scale

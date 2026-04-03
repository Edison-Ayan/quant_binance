"""
ShockDetector - 微观结构冲击检测（插针 / 流动性真空保护）

本质：插针不是普通波动，而是：
    - order book 被瞬间吃穿（depth collapse）
    - spread 骤然扩大
    - 单笔异常大单横扫
    - 1 秒内价格跳动超出正常范围

冲击判断（满足任意 2 条即认定为冲击）：
    C1. spread_zscore  > 3σ       买卖价差异常扩大
    C2. trade_size_mult > 5×      单笔成交额 = 过去均值的 5 倍以上
    C3. depth_ratio    < 50%      盘口深度骤降至均值一半以下
    C4. price_jump_1s  > 0.5%     1 秒内价格跳动超 0.5%

触发后：
    - 该品种进入 30 秒暂停窗口（禁止开仓）
    - 同时通知策略层紧急平仓

Kill Switch（全局）：
    60 秒内全市场冲击次数 ≥ 5 → 全局暂停 120 秒
    （连续插针行情直接停机，防止被连续打穿）
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── 检测阈值 ──────────────────────────────────────────────────────────────────
SPREAD_WINDOW        = 120    # spread 滚动窗口（bookTicker 更新笔数）
TRADE_SIZE_WINDOW    = 100    # 成交量滚动窗口
DEPTH_WINDOW         = 30     # 盘口深度滚动窗口
PRICE_TS_WINDOW      = 60     # 价格序列窗口（用于 1s 跳动检测）

SPREAD_ZSCORE_THRESH = 3.0    # spread Z-score 触发阈值
TRADE_SIZE_MULT      = 5.0    # 单笔大单倍数阈值（相对均值）
DEPTH_DROP_RATIO     = 0.50   # 深度骤降比例：低于均值 50% 触发
PRICE_JUMP_1S        = 0.010  # 1 秒价格跳动阈值（1.0%，加密货币 0.5% 过于灵敏）

SHOCK_CONDS_REQUIRED = 2      # 同时满足几个条件才认定为冲击

# Kill Switch 参数
KILL_WINDOW_SEC      = 60     # 滑动窗口（秒）
KILL_COUNT_THRESH    = 5      # 窗口内冲击次数触发 kill switch
KILL_PAUSE_SEC       = 60     # kill switch 全局暂停时长（秒，原 120s 占两个完整排名周期过长）
SYMBOL_PAUSE_SEC     = 30     # 单品种冲击后暂停时长（秒）


@dataclass
class ShockInfo:
    """冲击检测结果"""
    is_shocked:  bool       = False
    shock_score: int        = 0      # 触发条件数量（0~4）
    reasons:     List[str]  = field(default_factory=list)
    symbol:      str        = ""


class _SymbolState:
    """单个品种的微观结构状态缓冲区（内部使用）"""

    def __init__(self, symbol: str):
        self.symbol = symbol

        # spread 滚动统计（O(1) 更新）
        self._spread_buf:    deque = deque(maxlen=SPREAD_WINDOW)
        self._spread_sum:    float = 0.0
        self._spread_sum_sq: float = 0.0

        # 成交量滚动统计
        self._size_buf:    deque = deque(maxlen=TRADE_SIZE_WINDOW)
        self._size_sum:    float = 0.0

        # 盘口深度滚动统计
        self._depth_buf: deque = deque(maxlen=DEPTH_WINDOW)
        self._depth_sum: float = 0.0

        # 价格时间序列（用于 1s 跳动）
        self._price_ts: deque = deque(maxlen=PRICE_TS_WINDOW)

        # 暂停时间戳
        self._paused_until: float = 0.0

    # ── spread ────────────────────────────────────────────────────────────────

    def update_spread(self, spread_bps: float) -> Optional[float]:
        """返回当前 spread 的 Z-score（数据不足时返回 None）"""
        if len(self._spread_buf) == SPREAD_WINDOW:
            old = self._spread_buf[0]
            self._spread_sum    -= old
            self._spread_sum_sq -= old * old

        self._spread_buf.append(spread_bps)
        self._spread_sum    += spread_bps
        self._spread_sum_sq += spread_bps * spread_bps

        n = len(self._spread_buf)
        if n < 20:
            return None

        mu    = self._spread_sum / n
        var   = max(self._spread_sum_sq / n - mu * mu, 0.0)
        sigma = var ** 0.5
        return (spread_bps - mu) / sigma if sigma > 1e-10 else 0.0

    # ── trade size ────────────────────────────────────────────────────────────

    def update_trade_size(self, usdt_vol: float) -> Optional[float]:
        """返回当前成交额相对均值的倍数（数据不足时返回 None）"""
        if len(self._size_buf) == TRADE_SIZE_WINDOW:
            self._size_sum -= self._size_buf[0]

        self._size_buf.append(usdt_vol)
        self._size_sum += usdt_vol

        n = len(self._size_buf)
        if n < 10:
            return None

        mu = self._size_sum / n
        return usdt_vol / mu if mu > 1e-10 else 0.0

    # ── depth ─────────────────────────────────────────────────────────────────

    def update_depth(self, total_depth_usdt: float) -> Optional[float]:
        """
        返回当前盘口深度相对均值的比例（0~1）。
        比例 < 0.5 说明深度骤降一半。
        """
        if len(self._depth_buf) == DEPTH_WINDOW:
            self._depth_sum -= self._depth_buf[0]

        self._depth_buf.append(total_depth_usdt)
        self._depth_sum += total_depth_usdt

        n = len(self._depth_buf)
        if n < 5:
            return None

        avg = self._depth_sum / n
        return total_depth_usdt / avg if avg > 1e-10 else 1.0

    # ── price 1s jump ─────────────────────────────────────────────────────────

    def update_price(self, price: float, ts: float) -> Optional[float]:
        """返回过去 1 秒内的价格跳动幅度（None = 数据不足）

        断线重连保护：若当前时间戳与上一条记录间隔 > 5 秒，说明 WS 发生过断线，
        丢弃所有历史价格数据，避免用断线前的旧价格比较产生虚假跳动检测。
        """
        # 断线缺口检测：时间戳跳变 > 5s → 清空历史（WS 重连场景）
        if self._price_ts and ts - self._price_ts[-1][0] > 5.0:
            self._price_ts.clear()

        self._price_ts.append((ts, price))

        target_ts = ts - 1.0
        # 从最新往前找第一个 ≤ target_ts 的价格（1 秒前的参考价）
        series = list(self._price_ts)
        for i in range(len(series) - 2, -1, -1):
            t_i, p_i = series[i]
            if t_i <= target_ts and p_i > 0:
                return abs(price - p_i) / p_i
        return None

    # ── 暂停状态 ──────────────────────────────────────────────────────────────

    def pause(self, seconds: float = SYMBOL_PAUSE_SEC):
        self._paused_until = time.time() + seconds

    def is_paused(self) -> bool:
        return time.time() < self._paused_until


class ShockDetector:
    """
    全市场微观结构冲击检测器。

    使用方式：
        在 on_tick  → shock_detector.on_trade(...)
        在 on_book  → shock_detector.on_book(...)
        开仓前      → shock_detector.is_paused(symbol)
        全局检查    → shock_detector.is_kill_switched()
    """

    def __init__(self):
        self._states:            Dict[str, _SymbolState] = {}
        self._shock_timestamps:  deque                   = deque()  # 全局冲击记录
        self._kill_until:        float                   = 0.0

    # ── 内部 ─────────────────────────────────────────────────────────────────

    def _get(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(symbol)
        return self._states[symbol]

    def _eval(self, state: _SymbolState, reasons: List[str]) -> ShockInfo:
        """
        评估冲击条件数量，决定是否触发。
        触发时：暂停该品种、记录全局冲击时间戳、检查 kill switch。
        """
        score = len(reasons)
        if score < SHOCK_CONDS_REQUIRED:
            return ShockInfo(is_shocked=False, shock_score=score,
                             reasons=reasons, symbol=state.symbol)

        # ── 触发冲击 ──────────────────────────────────────────────────────
        state.pause(SYMBOL_PAUSE_SEC)

        now = time.time()
        self._shock_timestamps.append(now)

        # 清理过期记录
        cutoff = now - KILL_WINDOW_SEC
        while self._shock_timestamps and self._shock_timestamps[0] < cutoff:
            self._shock_timestamps.popleft()

        # Kill switch 判断
        if len(self._shock_timestamps) >= KILL_COUNT_THRESH:
            self._kill_until = now + KILL_PAUSE_SEC

        return ShockInfo(is_shocked=True, shock_score=score,
                         reasons=reasons, symbol=state.symbol)

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def on_trade(
        self,
        symbol:   str,
        price:    float,
        usdt_vol: float,
        ts_ms:    int,
    ) -> ShockInfo:
        """
        在每笔 aggTrade 到来时调用。

        检测：C2（大单）+ C4（1s 价格跳动）
        """
        ts    = ts_ms / 1000.0
        state = self._get(symbol)
        reasons: List[str] = []

        # C4：1 秒价格跳动
        jump = state.update_price(price, ts)
        if jump is not None and jump > PRICE_JUMP_1S:
            reasons.append(f"price_jump={jump:.3%}")

        # C2：异常大单
        size_mult = state.update_trade_size(usdt_vol)
        if size_mult is not None and size_mult > TRADE_SIZE_MULT:
            reasons.append(f"big_trade={size_mult:.1f}x")

        return self._eval(state, reasons)

    def on_book(
        self,
        symbol:    str,
        spread_bps: float,
        bid_usdt:  float,
        ask_usdt:  float,
    ) -> ShockInfo:
        """
        在每次 bookTicker 更新时调用。

        检测：C1（spread 扩大）+ C3（depth 骤降）

        参数：
            spread_bps : 当前买卖价差（基点）
            bid_usdt   : 最优买价 × 最优买量（USDT）
            ask_usdt   : 最优卖价 × 最优卖量（USDT）
        """
        state = self._get(symbol)
        reasons: List[str] = []

        # C1：spread 骤扩
        spread_z = state.update_spread(spread_bps)
        if spread_z is not None and spread_z > SPREAD_ZSCORE_THRESH:
            reasons.append(f"spread_z={spread_z:.1f}σ")

        # C3：盘口深度骤降
        total_depth = bid_usdt + ask_usdt
        depth_ratio = state.update_depth(total_depth)
        if depth_ratio is not None and depth_ratio < (1.0 - DEPTH_DROP_RATIO):
            reasons.append(f"depth_drop={1-depth_ratio:.0%}")

        return self._eval(state, reasons)

    def is_paused(self, symbol: str) -> bool:
        """该品种是否处于冲击后暂停窗口（禁止开仓）"""
        state = self._states.get(symbol)
        return state.is_paused() if state else False

    def is_kill_switched(self) -> bool:
        """全局 kill switch 是否触发（全市场暂停开仓）"""
        return time.time() < self._kill_until

    def get_status(self) -> dict:
        """返回当前状态摘要，供日志和监控使用"""
        now     = time.time()
        paused  = [s for s, st in self._states.items() if st.is_paused()]
        return {
            "kill_switched":  self.is_kill_switched(),
            "kill_resume_in": max(0.0, self._kill_until - now),
            "recent_shocks":  len(self._shock_timestamps),
            "paused_count":   len(paused),
            "paused_symbols": paused[:10],   # 只显示前 10 个，防止日志爆炸
        }

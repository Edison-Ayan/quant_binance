import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ── volume flow 参数（新版）───────────────────────────────────────────────────
# 当前净资金流统计窗口（秒）：统计最近多久的 signed volume
VOLUME_FLOW_WINDOW_SECONDS = 60

# volume flow zscore 的历史窗口（样本个数）
# 每次 on_trade 都会更新一次当前窗口总 signed volume，并记录到历史中
VOLUME_FLOW_HISTORY = 120

# 历史样本最少数量（不足时不输出有效 zscore）
MIN_VOLUME_HISTORY = 20

# 价格历史窗口：高频品种100tick/s时覆盖约20秒；低频时更长
PRICE_WINDOW = 2000

# OFI EMA 衰减系数（替代固定窗口 sum）
# 0.3 ≈ 等效 ~5 次更新的半衰期，对近期变化更敏感
OFI_EMA_ALPHA = 0.3

# 价格收益率回溯的时间窗口（秒）
RET_1M_SECONDS = 60
RET_5M_SECONDS = 300

# 价格收益率计算节流：每隔 N 笔成交才重算一次（降低 O(PRICE_WINDOW) 扫描频率）
RET_UPDATE_INTERVAL = 10


@dataclass
class SymbolFeatures:
    """
    单个币种的实时特征向量（打分引擎的输入）。

    所有字段默认为 0.0（数据不足时不参与打分或得分为0）。
    """
    symbol:           str

    # Flow 类
    volume_zscore:    float = 0.0   # 最近一段时间 signed volume 的异常程度（净买压=正，净卖压=负）
    ofi:              float = 0.0   # > 0 = 主动买占优；< 0 = 主动卖占优

    # 价格类
    ret_1m:           float = 0.0
    ret_5m:           float = 0.0

    # 流动性类
    spread_bps:       float = 0.0
    depth_imbalance:  float = 0.0
    best_depth_usdt:  float = 0.0

    # 衍生品类
    funding_rate:     float = 0.0
    oi_change_pct:    float = 0.0
    ret_24h:          float = 0.0

    # LOB 流形类（由 LOBManifoldEngine 填充）
    lob_pc1:          float = 0.0
    lob_z1:           float = 0.0
    lob_z2:           float = 0.0
    lob_z3:           float = 0.0
    lob_bucket:       str   = "mid"

    # 元数据
    last_price:       float = 0.0
    last_update:      float = 0.0
    data_count:       int   = 0


class SymbolState:
    """
    单个币种的实时数据缓冲区。
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

        # ── 新版 volume flow：按时间窗口累计 signed volume ────────────────────
        # 元素: (ts, signed_usdt_volume)
        self.signed_volume_events: deque = deque()
        self._signed_volume_sum: float = 0.0

        # 历史“当前窗口净资金流”的序列，用于做 zscore
        self.volume_flow_history: deque = deque(maxlen=VOLUME_FLOW_HISTORY)
        self._flow_hist_sum: float = 0.0
        self._flow_hist_sum_sq: float = 0.0

        # ── 价格时间序列 ──────────────────────────────────────────────
        self.price_series: deque = deque(maxlen=PRICE_WINDOW)
        self._ret_tick_counter: int = 0

        # ── OFI EMA 状态 ────────────────────────────────────────────
        self._ofi_ema: float = 0.0

        # ── bookTicker 上一次状态（用于计算 delta）──────────────────
        self._prev_bid:     float = 0.0
        self._prev_ask:     float = 0.0
        self._prev_bid_qty: float = 0.0
        self._prev_ask_qty: float = 0.0

        # ── 衍生品数据（REST 轮询更新）──────────────────────────────
        self._prev_oi: float = 0.0
        self._curr_oi: float = 0.0

        # ── 当前特征向量 ─────────────────────────────────────────────
        self.features = SymbolFeatures(symbol=symbol)

    # ─── 公开接口（由 FeatureEngine 调用）────────────────────────────────────

    def on_trade(self, price: float, qty: float, is_buyer_maker: bool, ts_ms: int):
        """
        处理 aggTrade 推送。

        参数：
            price          : 成交价格
            qty            : 成交数量（币本位）
            is_buyer_maker : True = 买方为做市方（主动卖）；False = 买方为吃单方（主动买）
            ts_ms          : 成交时间戳（毫秒）
        """
        ts = ts_ms / 1000.0
        usdt_volume = price * qty

        self.features.last_price  = price
        self.features.last_update = ts
        self.features.data_count += 1

        # 真实主动方向：
        # is_buyer_maker=False → taker buy → 正
        # is_buyer_maker=True  → taker sell → 负
        trade_sign = -1.0 if is_buyer_maker else 1.0
        signed_usdt_volume = usdt_volume * trade_sign

        # 新版 volume_zscore：按时间窗口累计净资金流，再做历史 zscore
        self._update_volume_zscore(ts, signed_usdt_volume)

        # 价格序列
        self.price_series.append((ts, price))

        # 收益率节流更新
        self._ret_tick_counter += 1
        if self._ret_tick_counter >= RET_UPDATE_INTERVAL:
            self._ret_tick_counter = 0
            self._update_returns(ts, price)

    def on_book_ticker(self, bid: float, bid_qty: float, ask: float, ask_qty: float):
        """
        处理 bookTicker 推送（最优买卖价快照）。
        """
        # ── 计算 OFI delta ──────────────────────────────────────────
        if bid > self._prev_bid:
            ofi_bid = bid_qty
        elif bid == self._prev_bid:
            ofi_bid = bid_qty - self._prev_bid_qty
        else:
            ofi_bid = -self._prev_bid_qty

        if ask < self._prev_ask:
            ofi_ask = -ask_qty
        elif ask == self._prev_ask:
            ofi_ask = self._prev_ask_qty - ask_qty
        else:
            ofi_ask = self._prev_ask_qty

        ofi_delta = ofi_bid - ofi_ask
        self._ofi_ema = OFI_EMA_ALPHA * ofi_delta + (1 - OFI_EMA_ALPHA) * self._ofi_ema
        self.features.ofi = self._ofi_ema

        # ── 计算价差和盘口失衡 ────────────────────────────────────────
        if ask > 0 and bid > 0:
            mid = (bid + ask) / 2.0
            self.features.spread_bps = (ask - bid) / mid * 10_000

        total_depth = bid_qty + ask_qty
        if total_depth > 0:
            self.features.depth_imbalance = (bid_qty - ask_qty) / total_depth

        # ── 真实 USDT 深度
        if bid > 0 and ask > 0:
            self.features.best_depth_usdt = bid * bid_qty + ask * ask_qty

        # ── 保存当前状态 ───────────────────────────────────────────────
        self._prev_bid     = bid
        self._prev_ask     = ask
        self._prev_bid_qty = bid_qty
        self._prev_ask_qty = ask_qty

    def update_derivatives(self, funding_rate: float, oi: float, ret_24h: float = 0.0):
        """
        更新资金费率、持仓量和24h涨跌幅（由 REST Fetcher 定期调用）。
        """
        self.features.funding_rate = funding_rate
        self.features.ret_24h = ret_24h

        if self._prev_oi > 0:
            oi_change = (oi - self._prev_oi) / self._prev_oi
            ret_sign = 1.0 if self.features.ret_1m > 0 else (-1.0 if self.features.ret_1m < 0 else 0.0)
            self.features.oi_change_pct = abs(oi_change) * ret_sign

        self._prev_oi = self._curr_oi
        self._curr_oi = oi

    # ─── 内部计算方法 ────────────────────────────────────────────────────────

    def _update_volume_zscore(self, ts: float, signed_usdt_volume: float):
        """
        新版 volume_zscore：

        1. 维护最近 VOLUME_FLOW_WINDOW_SECONDS 秒内的 signed volume 累积
        2. 将“当前窗口净资金流”记录到 history
        3. 对该净资金流做历史 zscore

        结果解释：
            > 0 : 最近一段时间净买压显著高于历史常态
            < 0 : 最近一段时间净卖压显著高于历史常态
        """
        # 1) 追加当前事件
        self.signed_volume_events.append((ts, signed_usdt_volume))
        self._signed_volume_sum += signed_usdt_volume

        # 2) 滚动剔除超出时间窗口的旧事件
        cutoff = ts - VOLUME_FLOW_WINDOW_SECONDS
        while self.signed_volume_events and self.signed_volume_events[0][0] < cutoff:
            _, old_sv = self.signed_volume_events.popleft()
            self._signed_volume_sum -= old_sv

        # 当前窗口净资金流
        current_flow = self._signed_volume_sum

        # 3) 把 current_flow 记入 history（滚动 O(1) 统计）
        if len(self.volume_flow_history) == VOLUME_FLOW_HISTORY:
            old = self.volume_flow_history[0]
            self._flow_hist_sum    -= old
            self._flow_hist_sum_sq -= old * old

        self.volume_flow_history.append(current_flow)
        self._flow_hist_sum    += current_flow
        self._flow_hist_sum_sq += current_flow * current_flow

        n = len(self.volume_flow_history)
        if n < MIN_VOLUME_HISTORY:
            self.features.volume_zscore = 0.0
            return

        mu = self._flow_hist_sum / n
        variance = self._flow_hist_sum_sq / n - mu * mu
        sigma = variance ** 0.5 if variance > 1e-20 else 0.0

        if sigma > 1e-10:
            z = (current_flow - mu) / sigma
            # 软一点的截断，防止极端事件把截面打穿
            if z > 3.0:
                z = 3.0
            elif z < -3.0:
                z = -3.0
            self.features.volume_zscore = z
        else:
            self.features.volume_zscore = 0.0

    def _update_returns(self, current_ts: float, current_price: float):
        """
        基于时间戳计算价格收益率（二分查找，O(log N) 每次调用）。
        """
        if len(self.price_series) < 2:
            return

        series = list(self.price_series)
        timestamps = [t for t, _ in series]

        def find_price_at_offset(seconds_ago: float) -> Optional[float]:
            target_ts = current_ts - seconds_ago
            lo, hi = 0, len(timestamps) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if timestamps[mid] < target_ts:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx > 0:
                if abs(timestamps[idx - 1] - target_ts) < abs(timestamps[idx] - target_ts):
                    idx = idx - 1
            best_diff = abs(timestamps[idx] - target_ts)
            if best_diff < seconds_ago * 0.5:
                return series[idx][1]
            return None

        p_1m = find_price_at_offset(RET_1M_SECONDS)
        if p_1m and p_1m > 0:
            self.features.ret_1m = (current_price - p_1m) / p_1m

        p_5m = find_price_at_offset(RET_5M_SECONDS)
        if p_5m and p_5m > 0:
            self.features.ret_5m = (current_price - p_5m) / p_5m


class FeatureEngine:
    """
    全市场特征引擎。
    """

    def __init__(self):
        self._states: Dict[str, SymbolState] = {}

    def _get_state(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            self._states[symbol] = SymbolState(symbol)
        return self._states[symbol]

    def on_trade(self, symbol: str, price: float, qty: float, is_buyer_maker: bool, ts_ms: int):
        self._get_state(symbol).on_trade(price, qty, is_buyer_maker, ts_ms)

    def on_book_ticker(self, symbol: str, bid: float, bid_qty: float, ask: float, ask_qty: float):
        self._get_state(symbol).on_book_ticker(bid, bid_qty, ask, ask_qty)

    def update_derivatives(self, symbol: str, funding_rate: float, oi: float, ret_24h: float = 0.0):
        self._get_state(symbol).update_derivatives(funding_rate, oi, ret_24h)

    def update_lob_latent(
        self,
        symbol: str,
        z: "np.ndarray",
        bucket: str = "mid",
        pc1: float = 0.0,
    ):
        f = self._get_state(symbol).features
        f.lob_pc1 = pc1
        if len(z) >= 1:
            f.lob_z1 = float(z[0])
        if len(z) >= 2:
            f.lob_z2 = float(z[1])
        if len(z) >= 3:
            f.lob_z3 = float(z[2])
        f.lob_bucket = bucket

    def get_all_features(self) -> Dict[str, SymbolFeatures]:
        return {
            sym: state.features
            for sym, state in self._states.items()
            if state.features.data_count >= 20
        }

    def get_state(self, symbol: str) -> Optional[SymbolState]:
        return self._states.get(symbol)

    @property
    def active_symbols(self) -> List[str]:
        return list(self._states.keys())
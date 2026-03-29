"""
Feature Engine - 实时多因子特征计算

职责：
    为每个币种维护一个独立的状态缓冲区（SymbolState），
    基于 WebSocket 推送的 aggTrade 和 bookTicker 数据，实时计算多维特征向量。

特征体系：

    Flow 类（最重要）：
        volume_zscore    - 成交量 Z-score × 价格方向（放量上涨=正，放量下跌=负）
        ofi              - 订单流失衡（买方主动成交量 - 卖方主动成交量）

    价格类：
        ret_1m           - 过去1分钟价格收益率（已涨了要扣分）
        ret_5m           - 过去5分钟价格收益率

    流动性类：
        spread_bps       - 买卖价差（基点），越小越容易成交
        depth_imbalance  - 盘口深度失衡 (bid_depth - ask_depth) / total

    衍生品类（由 REST 轮询更新）：
        funding_rate     - 当前资金费率（负值=空头拥挤=做多有利）
        oi_change_pct    - 持仓量变化率

计算方式：
    - 使用 deque(maxlen=N) 作为环形缓冲区，O(1) 插入，自动丢弃老数据
    - Volume Z-score 基于最近 WINDOW 笔成交滚动计算
    - OFI 基于 bookTicker 的买卖盘量变化累积（参考 Cont et al. 2014）
    - 价格收益率基于时间戳索引，精确控制回溯窗口
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# 滚动窗口大小（笔数）
VOLUME_WINDOW = 60       # 成交量 Z-score 窗口（最近60笔）
PRICE_WINDOW  = 2000     # 价格历史窗口：高频品种100tick/s时覆盖约20秒；低频时更长

# OFI EMA 衰减系数（替代固定窗口 sum）
# 0.3 ≈ 等效 ~5 次更新的半衰期，对近期变化更敏感
OFI_EMA_ALPHA = 0.3

# 价格收益率回溯的时间窗口（秒）
RET_1M_SECONDS = 60
RET_5M_SECONDS = 300

# 价格收益率计算节流：每隔 N 笔成交才重算一次（降低 O(PRICE_WINDOW) 扫描频率）
# 收益率本身变化缓慢，每10笔更新一次误差极小
RET_UPDATE_INTERVAL = 10


@dataclass
class SymbolFeatures:
    """
    单个币种的实时特征向量（打分引擎的输入）。

    所有字段默认为 0.0（数据不足时不参与打分或得分为0）。
    """
    symbol:           str

    # Flow 类
    volume_zscore:    float = 0.0   # |量级| × sign(ret_1m)：放量上涨=正，放量下跌=负
    ofi:              float = 0.0   # > 0 = 主动买占优；< 0 = 主动卖占优

    # 价格类
    ret_1m:           float = 0.0   # 最近1分钟收益率（已涨了要扣分）
    ret_5m:           float = 0.0   # 最近5分钟收益率

    # 流动性类
    spread_bps:       float = 0.0   # 买卖价差（基点）
    depth_imbalance:  float = 0.0   # > 0 = 买盘更厚 → 看涨；< 0 = 卖盘更厚 → 看跌

    # 衍生品类
    funding_rate:     float = 0.0   # 负值代表空头拥挤（做多有利）
    oi_change_pct:    float = 0.0   # OI 增加 + 价格上涨 = 趋势强化

    # LOB 流形类（由 LOBManifoldEngine 填充）
    lob_z1:           float = 0.0   # 第1主成分投影（最大方差方向，通常对应市场整体冲击）
    lob_z2:           float = 0.0   # 第2主成分投影（通常对应买卖不对称性）
    lob_z3:           float = 0.0   # 第3主成分投影（通常对应深度分布形状变化）
    lob_bucket:       str   = "mid" # 流动性桶（"high"/"mid"/"low"，由 LOBManifoldEngine 分配）

    # 元数据
    last_price:       float = 0.0   # 最新成交价（用于仓位计算）
    last_update:      float = 0.0   # 最后更新时间戳（Unix seconds）
    data_count:       int   = 0     # 已接收数据量（数据量不足时跳过打分）


class SymbolState:
    """
    单个币种的实时数据缓冲区。

    负责维护滚动窗口并计算各维度特征，每次收到 aggTrade 或 bookTicker 更新时触发。
    使用 deque(maxlen=N) 实现 O(1) 的环形缓冲区，无需手动管理窗口边界。
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

        # ── 成交量缓冲区 ──────────────────────────────────────────────
        self.trade_volumes: deque = deque(maxlen=VOLUME_WINDOW)
        # 滚动统计：维护 sum 和 sum_sq，避免每 tick 重建 numpy 数组（O(N)→O(1)）
        self._vol_sum:    float = 0.0
        self._vol_sum_sq: float = 0.0

        # ── 价格时间序列 ──────────────────────────────────────────────
        # (timestamp_seconds, price) 元组，用于精确时间窗口的收益率计算
        self.price_series: deque = deque(maxlen=PRICE_WINDOW)
        # 节流计数器：每 RET_UPDATE_INTERVAL 笔才重算收益率
        self._ret_tick_counter: int = 0

        # ── OFI EMA 状态 ────────────────────────────────────────────
        # 用指数衰减替代固定窗口 sum，对近期变化更敏感，无边界效应
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

        # 真实成交方向：is_buyer_maker=False 表示买方主动吃单（taker buy）→ 正
        trade_sign = -1.0 if is_buyer_maker else 1.0

        # 成交量 Z-score：滚动统计 O(1) 更新，方向来自成交主动方
        self._update_volume_zscore(usdt_volume, trade_sign)

        # 价格序列：始终追加（供收益率回溯使用）
        self.price_series.append((ts, price))

        # 收益率：节流计算，每 RET_UPDATE_INTERVAL 笔才扫描一次
        self._ret_tick_counter += 1
        if self._ret_tick_counter >= RET_UPDATE_INTERVAL:
            self._ret_tick_counter = 0
            self._update_returns(ts, price)

    def on_book_ticker(self, bid: float, bid_qty: float, ask: float, ask_qty: float):
        """
        处理 bookTicker 推送（最优买卖价快照）。

        OFI 计算公式（Cont, Kukanov, Stoikov 2014）：
            如果 bid 没变：OFI_bid = bid_qty - prev_bid_qty
            如果 bid 上升：OFI_bid = bid_qty
            如果 bid 下降：OFI_bid = -prev_bid_qty
            （ask 侧对称取负）

        直觉：
            买盘量增加 → 买方积极 → 上行压力
            卖盘量增加 → 卖方积极 → 下行压力
        """
        # ── 计算 OFI delta ──────────────────────────────────────────
        if bid > self._prev_bid:          # 买盘价格上移 → 新增买盘
            ofi_bid = bid_qty
        elif bid == self._prev_bid:       # 买盘价格不变 → 量的变化
            ofi_bid = bid_qty - self._prev_bid_qty
        else:                             # 买盘价格下移 → 买盘撤离
            ofi_bid = -self._prev_bid_qty

        if ask < self._prev_ask:          # 卖盘价格下移 → 新增卖盘
            ofi_ask = -ask_qty
        elif ask == self._prev_ask:
            ofi_ask = self._prev_ask_qty - ask_qty
        else:
            ofi_ask = self._prev_ask_qty  # 卖盘撤离 → 对多头有利

        ofi_delta = ofi_bid - ofi_ask
        # EMA 衰减：对近期变化更敏感，无固定窗口边界效应
        self._ofi_ema = OFI_EMA_ALPHA * ofi_delta + (1 - OFI_EMA_ALPHA) * self._ofi_ema
        self.features.ofi = self._ofi_ema

        # ── 计算价差和盘口失衡 ────────────────────────────────────────
        if ask > 0 and bid > 0:
            mid = (bid + ask) / 2.0
            self.features.spread_bps = (ask - bid) / mid * 10_000

        total_depth = bid_qty + ask_qty
        if total_depth > 0:
            self.features.depth_imbalance = (bid_qty - ask_qty) / total_depth

        # ── 保存当前状态 ───────────────────────────────────────────────
        self._prev_bid     = bid
        self._prev_ask     = ask
        self._prev_bid_qty = bid_qty
        self._prev_ask_qty = ask_qty

    def update_derivatives(self, funding_rate: float, oi: float):
        """
        更新资金费率和持仓量（由 REST Fetcher 定期调用）。

        OI 变化率含义：
            oi_change_pct > 0 且价格上涨 → 趋势强化（多头入场）
            oi_change_pct < 0 且价格上涨 → 可能是空头平仓（反弹动能有限）
        """
        self.features.funding_rate = funding_rate
        if self._prev_oi > 0:
            oi_change = (oi - self._prev_oi) / self._prev_oi
            # 方向化：OI 本身多空模糊，乘以价格方向才有意义
            # OI↑ + 价格↑ → 多头入场（正）；OI↑ + 价格↓ → 空头入场（负）
            ret_sign = 1.0 if self.features.ret_1m > 0 else (-1.0 if self.features.ret_1m < 0 else 0.0)
            self.features.oi_change_pct = abs(oi_change) * ret_sign
        self._prev_oi = self._curr_oi
        self._curr_oi = oi

    # ─── 内部计算方法 ────────────────────────────────────────────────────────

    def _update_volume_zscore(self, new_vol: float, trade_sign: float):
        """
        成交量 Z-score（滚动统计，O(1) 每 tick）。

        方向来自 is_buyer_maker（真实主动方），不再依赖滞后的 ret_1m：
            taker buy  (is_buyer_maker=False) → trade_sign = +1
            taker sell (is_buyer_maker=True)  → trade_sign = -1

        最终值 = |Z| × trade_sign：主动买放量=正，主动卖放量=负。
        """
        if len(self.trade_volumes) == VOLUME_WINDOW:
            old = self.trade_volumes[0]
            self._vol_sum    -= old
            self._vol_sum_sq -= old * old

        self.trade_volumes.append(new_vol)
        self._vol_sum    += new_vol
        self._vol_sum_sq += new_vol * new_vol

        n = len(self.trade_volumes)
        if n < 10:
            return

        mu       = self._vol_sum / n
        variance = self._vol_sum_sq / n - mu * mu
        sigma    = variance ** 0.5 if variance > 1e-20 else 0.0

        if sigma > 1e-10:
            magnitude = abs((new_vol - mu) / sigma)
            self.features.volume_zscore = magnitude * trade_sign
        else:
            self.features.volume_zscore = 0.0

    def _update_returns(self, current_ts: float, current_price: float):
        """
        基于时间戳计算价格收益率（二分查找，O(log N) 每次调用）。

        price_series 按时间戳单调递增追加，可直接用二分查找定位目标时刻。
        本方法经节流后每 RET_UPDATE_INTERVAL 笔成交才调用一次。
        """
        if len(self.price_series) < 2:
            return

        # 将 deque 转为 list 供二分查找（仅在节流触发时执行，开销可接受）
        series = list(self.price_series)
        timestamps = [t for t, _ in series]

        def find_price_at_offset(seconds_ago: float) -> Optional[float]:
            target_ts = current_ts - seconds_ago
            # 二分查找最接近 target_ts 的位置
            lo, hi = 0, len(timestamps) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if timestamps[mid] < target_ts:
                    lo = mid + 1
                else:
                    hi = mid
            # lo 是第一个 >= target_ts 的位置，比较左右两侧取更近的
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

    统一管理所有品种的 SymbolState，对外提供简洁的接口：
    - on_trade()           : 处理 aggTrade 推送
    - on_book_ticker()     : 处理 bookTicker 推送
    - update_derivatives() : 更新资金费率 / OI
    - get_all_features()   : 获取所有品种当前特征快照
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

    def update_derivatives(self, symbol: str, funding_rate: float, oi: float):
        self._get_state(symbol).update_derivatives(funding_rate, oi)

    def update_lob_latent(self, symbol: str, z: "np.ndarray", bucket: str = "mid"):
        """
        更新单个品种的 LOB 流形潜在特征（由 LOBManifoldEngine 调用）。

        参数：
            z      : np.ndarray(n_comp,) — 白化 PCA 投影坐标
            bucket : 流动性桶名称（"high"/"mid"/"low"）
        """
        f = self._get_state(symbol).features
        if len(z) >= 1:
            f.lob_z1 = float(z[0])
        if len(z) >= 2:
            f.lob_z2 = float(z[1])
        if len(z) >= 3:
            f.lob_z3 = float(z[2])
        f.lob_bucket = bucket

    def get_all_features(self) -> Dict[str, SymbolFeatures]:
        """
        获取所有品种的特征快照（过滤数据量不足的品种）。

        只返回 data_count >= 20 的品种，避免冷启动阶段的噪音信号。
        """
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

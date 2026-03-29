"""
毫秒级实时订单簿（MsOrderBookEngine）

═══════════════════════════════════════════════════════════════════════════════
与现有 OrderBookEngine 的核心差异
═══════════════════════════════════════════════════════════════════════════════

    OrderBookEngine（旧）            MsOrderBookEngine（本模块）
    ─────────────────────────────    ─────────────────────────────────────────
    现货端点（api.binance.com）      合约端点（fapi.binance.com）
    @depth@100ms（100ms 批处理）     @depth@100ms（100ms，10次/秒，最快可用）
    无时间戳追踪                     提取 E 字段（事件时间 ms）
    无数据清洗                       6 步清洗流水线
    无序列断链恢复                   自动重新初始化快照
    无质量指标                       DataQualityMetrics 实时统计
    无限频率事件                     事件节流（最小 20ms 间隔）

═══════════════════════════════════════════════════════════════════════════════
6 步数据清洗流水线
═══════════════════════════════════════════════════════════════════════════════

    Step 1  序列验证（Sequence Validation）
        检查 WebSocket 消息的 U（firstUpdateId）和 u（finalUpdateId）
        是否与本地 last_update_id 连续。
        ─ 过期（u ≤ last_update_id）  : 丢弃，无副作用
        ─ 连续（U ≤ last+1 ≤ u）      : 接受
        ─ 断链（U > last+1）          : 触发重新初始化，gap_count++

    Step 2  基础字段校验（Field Validation）
        ─ 价格 ≤ 0 或数量 < 0        : 丢弃该价位，invalid_level_count++
        ─ 消息时间戳回退              : 丢弃整条消息

    Step 3  增量更新应用（Incremental Update）
        qty > 0 → 插入/覆盖对应价位
        qty == 0 → 删除该价位（Binance 约定：0 表示清空）

    Step 4  穿越盘口检测（Crossed Market Detection）
        best_bid ≥ best_ask 属于数据异常（正常市场永远 bid < ask）
        ─ 发生时记录 crossed_count，跳过本次事件发布，等待下次更新修正

    Step 5  价格离群检测（Outlier Price Detection）
        距中间价超过 max_spread_pct（默认 5%）的价位视为异常
        ─ 不从本地订单簿删除（可能是合法的深层挂单），但 ORDER_BOOK 事件中不包含这些档位

    Step 6  限频发布（Rate-Limited Event Emission）
        连续高频更新时（高波动期间每秒可能有数百次更新），
        限制 ORDER_BOOK 事件的发出频率为最多 50次/秒（min_interval_ms = 20ms）。
        内部订单簿仍以原始频率更新，只是减少下游策略的回调频率。

═══════════════════════════════════════════════════════════════════════════════
Binance 合约 @depth 消息字段说明
═══════════════════════════════════════════════════════════════════════════════

    {
        "e": "depthUpdate",   # 事件类型
        "E": 1234567890123,   # 事件时间（毫秒）←── 毫秒精度时间戳
        "T": 1234567890100,   # 撮合引擎时间（毫秒，比 E 更精确）
        "s": "BTCUSDT",       # 交易对
        "U": 123456789,       # firstUpdateId（本批次第一个更新的序号）
        "u": 123456790,       # finalUpdateId（本批次最后一个更新的序号）
        "pu": 123456788,      # prevFinalUpdateId（上一批次的 finalUpdateId，用于连续性校验）
        "b": [["50000.0","1.5"], ...],  # 买盘变化
        "a": [["50001.0","2.0"], ...]   # 卖盘变化
    }

    注意：合约 @depth 消息含 "pu" 字段（prevFinalUpdateId），
    连续性校验应检查：
        pu == last_update_id  AND  U <= last_update_id + 1 <= u
"""

import time
import json
import heapq
import threading
import requests
import websocket
from collections import deque
from dataclasses import dataclass, field

from core.event import Event
from core.constants import EventType
from data_layer.logger import logger


# ══════════════════════════════════════════════════════════════════════════════
# 数据质量指标
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityMetrics:
    """
    实时追踪订单簿数据质量。

    所有字段均可从外部读取，用于监控面板展示和告警触发。
    quality_score() 综合所有指标，输出 [0, 1] 的质量评分。
    """

    # 总收到的 WebSocket 消息数
    total_messages:       int = 0
    # 过期消息数（u ≤ last_update_id，正常数量级较多，无需告警）
    stale_messages:       int = 0
    # 序列断链次数（触发重初始化，越多说明网络越差）
    gap_count:            int = 0
    # 因断链触发的完整重初始化次数
    reinit_count:         int = 0
    # 穿越盘口次数（bid ≥ ask，数据异常）
    crossed_count:        int = 0
    # 无效价位数（价格≤0 或数量<0）
    invalid_level_count:  int = 0
    # 因限频跳过的事件发布次数
    throttled_count:      int = 0
    # 成功发布的 ORDER_BOOK 事件数
    events_emitted:       int = 0
    # 最后收到消息的时间戳（ms），用于检测 feed 是否 stale
    last_message_ts_ms:   int = 0

    def quality_score(self) -> float:
        """
        综合质量评分 ∈ [0.0, 1.0]。

        计算方式：
            以 gap 和 crossed 为主要惩罚项（权重×10），
            invalid_level 为次要惩罚（权重×1），
            除以总消息数归一化。

        0.95+ : 优质数据，可放心使用
        0.90+ : 良好数据，轻微噪声
        0.80+ : 一般，建议监控
        <0.80 : 数据质量差，策略需谨慎
        """
        if self.total_messages == 0:
            return 1.0
        penalty = (self.gap_count * 10 +
                   self.crossed_count * 10 +
                   self.invalid_level_count * 1 +
                   self.reinit_count * 20)
        return max(0.0, 1.0 - penalty / max(self.total_messages, 1))

    def to_dict(self) -> dict:
        return {
            "total_messages":      self.total_messages,
            "stale_messages":      self.stale_messages,
            "gap_count":           self.gap_count,
            "reinit_count":        self.reinit_count,
            "crossed_count":       self.crossed_count,
            "invalid_level_count": self.invalid_level_count,
            "throttled_count":     self.throttled_count,
            "events_emitted":      self.events_emitted,
            "quality_score":       round(self.quality_score(), 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 毫秒级实时订单簿引擎
# ══════════════════════════════════════════════════════════════════════════════

class MsOrderBookEngine:
    """
    Binance 合约毫秒级实时订单簿引擎。

    数据来源：Binance Futures WebSocket @depth（实时增量流，无 100ms 批处理延迟）
    REST 快照：fapi.binance.com/fapi/v1/depth（合约端点）

    主要功能：
        1. 维护精确的本地 L2 订单簿（全量，不限深度）
        2. 6 步数据清洗流水线（序列校验→字段校验→更新→穿越检测→离群→限频）
        3. 序列断链时自动重新初始化（不丢失交易机会）
        4. 以 ms 级精度追踪每次更新的事件时间
        5. 向 EventEngine 发布 ORDER_BOOK 事件（限频 50次/秒）
        6. 实时维护 DataQualityMetrics

    使用示例：
        order_book = MsOrderBookEngine("BTCUSDT", event_engine, top_n=5)
        # 之后 ORDER_BOOK 事件自动流入 EventEngine，策略通过 on_order_book() 消费
    """

    # Binance 合约深度接口
    _REST_URL = "https://fapi.binance.com/fapi/v1/depth"
    _WS_URL   = "wss://fstream.binance.com/ws/{symbol}@depth@100ms"

    def __init__(self, symbol: str, event_engine=None,
                 top_n: int = 10,
                 min_event_interval_ms: int = 20,
                 max_price_spread_pct: float = 0.05,
                 stale_timeout_s: float = 10.0):
        """
        参数：
            symbol                (str)   : 合约交易对，例如 "BTCUSDT"
            event_engine                  : 全局事件引擎；None 则只维护本地状态
            top_n                 (int)   : ORDER_BOOK 事件携带的档位数（默认 5）
            min_event_interval_ms (int)   : ORDER_BOOK 事件最小发布间隔（ms），
                                            默认 20ms（即最多 50次/秒），
                                            防止高波动期间事件洪水淹没策略
            max_price_spread_pct  (float) : 价格离群阈值（相对中间价的最大偏离比例）
                                            超出此范围的价位在事件中会被过滤掉
                                            默认 5%（正常市场档位不会偏离这么远）
            stale_timeout_s       (float) : Feed 超时检测阈值（秒）
                                            超过此时间无消息则认为 feed 已断开
        """
        self.symbol      = symbol.upper()
        self._symbol_lc  = symbol.lower()
        self.event_engine = event_engine
        self.top_n       = top_n
        self._min_interval_ms  = min_event_interval_ms
        self._max_spread_pct   = max_price_spread_pct
        self._stale_timeout_s  = stale_timeout_s

        # L2 订单簿（全量，无深度限制）
        self.bids: dict = {}
        self.asks: dict = {}

        # 序列管理
        self.last_update_id: int = 0  # 本地已处理的最新 u 值

        # ms 时间戳追踪
        self.last_event_time_ms:    int = 0   # 最近一条 WS 消息的 E 字段
        self.last_match_time_ms:    int = 0   # 最近一条 WS 消息的 T 字段（撮合引擎时间）
        self._last_emitted_ms:      int = 0   # 最近一次发布 ORDER_BOOK 事件的时间（ms）

        # 数据质量指标
        self.metrics = DataQualityMetrics()

        # WebSocket 管理
        self._ws:          websocket.WebSocketApp = None
        self._ws_thread:   threading.Thread       = None
        self._running:     bool                   = False

        # 重初始化锁：防止 gap 处理时多次并发重初始化
        self._reinit_lock = threading.Lock()
        self._reinit_in_progress = False
        self._reinit_requested = False

        # Binance 推荐初始化流程：先订阅 WS（缓冲消息），再拉快照，再对齐
        self._message_buffer: deque = deque(maxlen=10000)
        self._initialized: bool = False
        self._ws_connected = threading.Event()

        # 步骤①：先建 WS 连接（此时消息进缓冲区）
        self._start_ws()
        if not self._ws_connected.wait(timeout=10.0):
            raise RuntimeError(f"[MsOrderBook:{self.symbol}] WebSocket 连接超时")
        time.sleep(0.05)  # 稍等，确保缓冲区积累了几条消息

        # 步骤②：拉 REST 快照
        self._init_snapshot()

        # 步骤③：从缓冲区找同步点，对齐本地账本
        self._sync_buffer()

    # ─── REST 快照初始化 ────────────────────────────────────────────────────────

    def _init_snapshot(self):
        """
        从 Binance 合约 REST API 获取完整深度快照。

        合约端点：GET fapi/v1/depth
            limit 可选 5/10/20/50/100/500/1000
            返回 lastUpdateId + bids + asks

        快照拉取后，last_update_id 设为快照的 lastUpdateId，
        后续 WebSocket 消息只有满足 U ≤ last_update_id+1 ≤ u 才被接受。
        """
        try:
            resp = requests.get(
                self._REST_URL,
                params={"symbol": self.symbol, "limit": 1000},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            logger.error(f"[MsOrderBook:{self.symbol}] REST 快照失败: {e}")
            raise

        self.last_update_id = data["lastUpdateId"]
        self.bids.clear()
        self.asks.clear()

        for p, q in data["bids"]:
            price, qty = float(p), float(q)
            if price > 0 and qty >= 0:
                if qty > 0:
                    self.bids[price] = qty

        for p, q in data["asks"]:
            price, qty = float(p), float(q)
            if price > 0 and qty >= 0:
                if qty > 0:
                    self.asks[price] = qty

        logger.info(f"[MsOrderBook:{self.symbol}] 快照初始化完成 "
                    f"lastUpdateId={self.last_update_id} "
                    f"bids={len(self.bids)} asks={len(self.asks)}")

    def _reinit_async(self):
        """
        在独立线程中重新初始化订单簿。

        设计原则：
            - 调用时立即进入缓冲模式（_initialized=False），确保新消息进缓冲区
            - 若已有 reinit 在运行，设置请求标志后直接返回，当前 reinit 结束后自动触发
            - 使用单线程阻塞轮询替代递归线程，彻底消除并发状态竞争
        """
        # 立即进入缓冲模式，阻止消息走正常处理路径
        self._initialized = False
        self._message_buffer.clear()

        with self._reinit_lock:
            if self._reinit_in_progress:
                self._reinit_requested = True   # 通知当前 reinit 结束后再来一次
                return
            self._reinit_in_progress = True
            self._reinit_requested = False

        def _do_reinit():
            try:
                # 等待 1s，让缓冲区积累足够消息（高延迟网络下必须足够长）
                time.sleep(1.0)
                self._init_snapshot()
                self._sync_buffer()
                self.metrics.reinit_count += 1
            except Exception as e:
                logger.error(f"[MsOrderBook:{self.symbol}] 重初始化失败: {e}")
                self._initialized = True  # 保底
            finally:
                with self._reinit_lock:
                    self._reinit_in_progress = False
                    requested = self._reinit_requested
                    self._reinit_requested = False
                # 若期间又有 reinit 请求，链式触发
                if requested:
                    self._reinit_async()

        threading.Thread(target=_do_reinit, daemon=True).start()

    def _sync_buffer(self):
        """
        单线程阻塞轮询，在超时时间内寻找缓冲区同步点。

        改进点（相比递归线程版本）：
            - 在同一线程内轮询，不产生并发副作用
            - _initialized 在整个轮询期间保持 False，消息稳定进入缓冲区
            - 超时后直接用快照状态启动（pu 连续性检查兜底后续 gap）
        """
        deadline    = time.time() + 5.0   # 最多等 5 秒
        poll_num    = 0

        while time.time() < deadline:
            # 读出当前所有缓冲消息（不清空，边读边判断）
            buffered = list(self._message_buffer)
            self._message_buffer.clear()

            logger.info(f"[MsOrderBook:{self.symbol}] 缓冲区同步 poll={poll_num}："
                        f"{len(buffered)} 条消息 lastUpdateId={self.last_update_id}")

            for i, raw in enumerate(buffered):
                msg = self._try_parse(raw)
                if msg is None:
                    continue

                u       = msg.get("u", 0)
                U_first = msg.get("U", 0)

                # 过期消息（快照已覆盖）
                if u <= self.last_update_id:
                    continue

                # 同步条件：U <= lastUpdateId+1 <= u
                if U_first <= self.last_update_id + 1:
                    logger.info(f"[MsOrderBook:{self.symbol}] 同步成功 "
                                f"U={U_first} u={u} lastUpdateId={self.last_update_id}")
                    remaining = [self._try_parse(r) for r in buffered[i + 1:]]
                    for sync_msg in [msg] + [m for m in remaining if m]:
                        self._apply_delta(sync_msg)
                        self.last_update_id = sync_msg.get("u", self.last_update_id)
                    self._initialized = True
                    return

            poll_num += 1
            time.sleep(0.2)   # 等 200ms 让更多消息进缓冲区

        # 超时：直接以快照状态启动，pu 连续性检查会捕捉后续 gap 并重新触发 reinit
        logger.warning(f"[MsOrderBook:{self.symbol}] 缓冲区同步超时（5s），"
                       f"以快照状态直接启动")
        self._initialized = True

    @staticmethod
    def _try_parse(raw: str):
        """JSON 解析辅助，失败返回 None。"""
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ─── WebSocket ─────────────────────────────────────────────────────────────

    def _start_ws(self):
        """
        建立 Binance 合约实时深度 WebSocket 连接。

        使用 @depth@100ms 以获得 100ms 间隔的增量推送（10次/秒）。
        合约 @depth 无后缀默认 250ms（4次/秒），@100ms 是最快的可用速率。
        """
        url = self._WS_URL.format(symbol=self._symbol_lc)
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        self._running = True
        self._ws_thread = threading.Thread(
            target=lambda: self._ws.run_forever(
                ping_interval=30, ping_timeout=20
            ),
            daemon=True
        )
        self._ws_thread.start()

    def _on_open(self, ws):
        logger.info(f"[MsOrderBook:{self.symbol}] WebSocket 连接已建立 (@depth 实时流)")
        self._ws_connected.set()

    def _on_error(self, ws, error):
        logger.error(f"[MsOrderBook:{self.symbol}] WebSocket 错误: {error}")

    def _on_close(self, ws, code, msg):
        logger.warning(f"[MsOrderBook:{self.symbol}] WebSocket 关闭 code={code}")
        if self._running:
            # 必须先进入缓冲模式，再建新连接，否则新 WS 的消息直接走正常处理路径
            # 不进缓冲区，导致 _sync_buffer 20 次重试全为空
            self._initialized = False
            self._message_buffer.clear()
            time.sleep(3)
            logger.info(f"[MsOrderBook:{self.symbol}] 重连中...")
            self._reinit_async()   # 重连前拉新快照
            self._start_ws()

    # ─── 消息处理主流程（6 步清洗流水线）──────────────────────────────────────────

    def _on_message(self, ws, raw_message: str):
        """
        WebSocket 消息回调：完整的 6 步数据清洗流水线。
        """
        self.metrics.total_messages += 1

        # 初始化同步期间：缓冲所有消息，等待 _sync_buffer() 处理
        if not self._initialized:
            self._message_buffer.append(raw_message)
            return

        # ── 解析 JSON ──────────────────────────────────────────────────────────
        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.warning(f"[MsOrderBook:{self.symbol}] JSON 解析失败: {e}")
            return

        # ── Step 1: 序列验证 ──────────────────────────────────────────────────
        first_update_id = msg.get("U", 0)
        final_update_id = msg.get("u", 0)
        prev_final_id   = msg.get("pu", 0)  # 合约特有字段：上一批次 finalUpdateId

        event_time_ms = msg.get("E", 0)
        match_time_ms = msg.get("T", 0)

        # 更新 ms 时间戳
        self.last_event_time_ms = event_time_ms
        self.last_match_time_ms = match_time_ms
        self.metrics.last_message_ts_ms = event_time_ms

        # 过期消息：该批次所有更新都已被快照覆盖
        if final_update_id <= self.last_update_id:
            self.metrics.stale_messages += 1
            return

        # 序列断链检测：
        # 对于合约 @depth 流，要求 pu == last_update_id（严格连续性）
        # 初始化阶段（last_update_id == 0）跳过此检查
        if (self.last_update_id > 0 and
                prev_final_id != 0 and
                prev_final_id != self.last_update_id):
            logger.warning(
                f"[MsOrderBook:{self.symbol}] 序列断链 "
                f"pu={prev_final_id} last={self.last_update_id}，触发重初始化"
            )
            self.metrics.gap_count += 1
            self._initialized = False       # 重新进入缓冲模式
            self._message_buffer.clear()    # 清空旧缓冲，从新快照同步
            self._reinit_async()
            return

        # 消息时间戳回退检测（防止乱序消息）
        if (event_time_ms > 0 and
                self.metrics.last_message_ts_ms > 0 and
                event_time_ms < self.metrics.last_message_ts_ms - 5000):
            # 允许 5 秒的时钟偏差容忍（网络路由可能导致轻微乱序）
            logger.warning(f"[MsOrderBook:{self.symbol}] 消息时间戳回退，丢弃")
            return

        # ── Step 2+3: 字段校验 + 增量更新应用 ─────────────────────────────────
        self._apply_delta(msg)

        # 推进序列号
        self.last_update_id = final_update_id

        # ── Step 4: 穿越盘口检测 ───────────────────────────────────────────────
        if self.bids and self.asks:
            best_bid = max(self.bids)
            best_ask = min(self.asks)
            if best_bid >= best_ask:
                self.metrics.crossed_count += 1
                logger.warning(
                    f"[MsOrderBook:{self.symbol}] 穿越盘口 "
                    f"bid={best_bid} >= ask={best_ask}，跳过本次事件"
                )
                return   # 不发布异常状态的盘口数据

        # ── Steps 5+6: 离群过滤 + 限频发布 ────────────────────────────────────
        self._emit_order_book(event_time_ms)

    def _apply_delta(self, msg: dict):
        """
        Step 2+3：字段校验 + 将增量变化应用到本地 L2 订单簿。

        校验规则：
            - 价格必须 > 0（负价格或零价格属于数据错误）
            - 数量必须 ≥ 0（负数量无意义）
            qty == 0 → 删除该价位（Binance 约定）
            qty > 0  → 插入/覆盖
        """
        for p_str, q_str in msg.get("b", []):
            try:
                price, qty = float(p_str), float(q_str)
            except (ValueError, TypeError):
                self.metrics.invalid_level_count += 1
                continue

            if price <= 0 or qty < 0:
                self.metrics.invalid_level_count += 1
                continue

            if qty == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for p_str, q_str in msg.get("a", []):
            try:
                price, qty = float(p_str), float(q_str)
            except (ValueError, TypeError):
                self.metrics.invalid_level_count += 1
                continue

            if price <= 0 or qty < 0:
                self.metrics.invalid_level_count += 1
                continue

            if qty == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    # ─── Step 5+6: 离群过滤 + 限频 ORDER_BOOK 事件发布 ──────────────────────────

    def _emit_order_book(self, event_time_ms: int):
        """
        Step 5: 过滤价格离群的档位。
        Step 6: 以限频方式发布 ORDER_BOOK 事件。

        参数：
            event_time_ms (int) : 本次更新的 Binance 事件时间（毫秒）
        """
        if self.event_engine is None or not self.bids or not self.asks:
            return

        # ── Step 6: 限频检查 ──────────────────────────────────────────────────
        now_ms = int(time.time() * 1000)
        if now_ms - self._last_emitted_ms < self._min_interval_ms:
            self.metrics.throttled_count += 1
            return

        # ── Step 5: 提取 top_n 档 ────────────────────────────────────────────
        # heapq.nlargest/nsmallest 是 O(n log k)，远优于 sorted() 的 O(n log n)
        # k=top_n=5 时，1000 档订单簿只需 O(n log 5) ≈ O(n) 次比较
        top_bids = heapq.nlargest(self.top_n, self.bids.items(), key=lambda x: x[0])
        top_asks = heapq.nsmallest(self.top_n, self.asks.items(), key=lambda x: x[0])

        if not top_bids or not top_asks:
            return

        best_bid_p, best_bid_q = top_bids[0]
        best_ask_p, best_ask_q = top_asks[0]
        mid = (best_bid_p + best_ask_p) / 2.0

        # ── Step 5: 价格离群过滤 ──────────────────────────────────────────────
        # 过滤距离中间价超过 max_price_spread_pct 的异常档位
        threshold = mid * self._max_spread_pct

        filtered_bids = [[p, q] for p, q in top_bids if abs(p - mid) <= threshold]
        filtered_asks = [[p, q] for p, q in top_asks if abs(p - mid) <= threshold]

        if not filtered_bids or not filtered_asks:
            return

        # ── 发布 ORDER_BOOK 事件 ───────────────────────────────────────────────
        event = Event(
            EventType.ORDER_BOOK,
            {
                "symbol":          self.symbol,
                "event_time_ms":   event_time_ms,   # Binance 推送时间（ms）
                "match_time_ms":   self.last_match_time_ms,  # 撮合引擎时间（ms）
                "local_time_ms":   now_ms,          # 本地接收时间（ms）
                "bids":            filtered_bids,
                "asks":            filtered_asks,
                "best_bid":        best_bid_p,
                "best_ask":        best_ask_p,
                "best_bid_qty":    best_bid_q,
                "best_ask_qty":    best_ask_q,
                "mid":             mid,
                "spread":          best_ask_p - best_bid_p,
                "quality_score":   self.metrics.quality_score(),  # 供策略参考数据质量
                # 兼容旧字段（timestamp）
                "timestamp":       event_time_ms,
            }
        )

        self.event_engine.put(event)
        self._last_emitted_ms = now_ms
        self.metrics.events_emitted += 1

    # ─── 公共查询接口（与旧 OrderBookEngine 兼容）──────────────────────────────

    def best_bid(self) -> float:
        """返回当前买一价，订单簿为空时返回 None。"""
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float:
        """返回当前卖一价，订单簿为空时返回 None。"""
        return min(self.asks) if self.asks else None

    def mid_price(self) -> float:
        """返回中间价，任意一侧为空时返回 None。"""
        bid, ask = self.best_bid(), self.best_ask()
        return (bid + ask) / 2.0 if bid and ask else None

    def spread(self) -> float:
        """返回买卖价差，订单簿不完整时返回 None。"""
        bid, ask = self.best_bid(), self.best_ask()
        return ask - bid if bid and ask else None

    def is_feed_stale(self) -> bool:
        """
        检测数据 feed 是否已停止更新（可能是 WebSocket 静默断线）。

        stale_timeout_s 秒内无任何消息到达则判定为 stale。
        """
        if self.metrics.last_message_ts_ms == 0:
            return False
        elapsed_s = (time.time() * 1000 - self.metrics.last_message_ts_ms) / 1000.0
        return elapsed_s > self._stale_timeout_s

    def get_quality_report(self) -> dict:
        """返回完整的数据质量报告，供监控面板和告警使用。"""
        report = self.metrics.to_dict()
        report.update({
            "symbol":         self.symbol,
            "last_update_id": self.last_update_id,
            "event_lag_ms":   int(time.time() * 1000) - self.metrics.last_message_ts_ms,
            "is_feed_stale":  self.is_feed_stale(),
            "bid_levels":     len(self.bids),
            "ask_levels":     len(self.asks),
            "best_bid":       self.best_bid(),
            "best_ask":       self.best_ask(),
            "mid_price":      self.mid_price(),
        })
        return report

    def close(self):
        """优雅关闭 WebSocket 连接，停止重连。"""
        self._running = False
        if self._ws:
            # websocket-client 的 close() 会在内部调用 recv_frame() 等待服务端关闭帧，
            # 而 _ws_thread 同时也在 recv()，两线程竞争同一 socket 会造成死锁。
            # 解决方案：在独立 daemon 线程中执行 close()，主线程最多等待 2s 后继续。
            ws = self._ws
            t = threading.Thread(target=ws.close, daemon=True, name="ws-close")
            t.start()
            t.join(timeout=2.0)

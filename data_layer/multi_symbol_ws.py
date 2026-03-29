"""
Multi-Symbol WebSocket Feed - 多币种实时数据订阅

职责：
    使用 Binance Futures Combined Stream 同时订阅多个交易对的实时数据，
    比单独连接每个品种效率高得多（一个连接 = 200个流）。

订阅的流类型：
    {symbol}@aggTrade       : 聚合成交（价格、数量、方向）
                              → 驱动 FeatureEngine.on_trade()
    {symbol}@bookTicker     : 最优买卖价快照（bid/ask + 数量）
                              → 驱动 FeatureEngine.on_book_ticker()
    {symbol}@depth5@100ms   : Top-5 档订单簿快照（100ms 间隔，完整快照非增量）
                              → 驱动 LOBManifoldEngine.on_order_book()（可选）

Combined Stream URL 格式：
    wss://fstream.binance.com/stream?streams=btcusdt@aggTrade/ethusdt@aggTrade/...

数据量估算（参考）：
    80 个品种 × 2 种流 = 160 个流
    aggTrade: ~10-100 条/秒（主流币更频繁）
    bookTicker: ~1-10 条/秒

Binance 限制：
    每个 WebSocket 连接最多 200 个流
    → 60 品种 × 2 流 = 120 流（无 depth）
    → 60 品种 × 3 流 = 180 流（含 depth5），仍在 200 上限内

自动重连：
    断线后自动在 RECONNECT_DELAY 秒后重连，保证数据连续性。
"""

import json
import os
import queue
import time
import threading
from typing import List, Callable, Optional
from urllib.parse import urlparse

import websocket

from data_layer.logger import logger


RECONNECT_DELAY    = 3    # 断线重连等待时间（秒）
MAX_STREAMS_PER_WS = 200  # Binance 单连接流数上限
SYMBOLS_PER_CONN   = 60   # 每个连接订阅的品种数（×2流=120，留40%余量）
# 消息分发队列容量：WS 线程只做 JSON 解析和入队，业务回调在独立线程执行，
# 避免回调慢时阻塞 WS recv 缓冲导致延迟积压。
DISPATCH_QUEUE_SIZE = 20_000


def _parse_proxy():
    """从环境变量读取代理配置（支持 HTTP_PROXY / HTTPS_PROXY）。"""
    raw = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
    if not raw:
        return {}
    p = urlparse(raw)
    return {
        "http_proxy_host": p.hostname,
        "http_proxy_port": p.port,
        "proxy_type":      "http",
    }


class MultiSymbolFeed:
    """
    Binance Futures Combined Stream 多币订阅客户端。

    将 aggTrade 和 bookTicker 推送解析后，
    分别回调 on_agg_trade 和 on_book_ticker 函数。
    """

    BASE_URL = "wss://fstream.binance.com/stream"

    def __init__(
        self,
        symbols:         List[str],
        on_agg_trade:    Callable[[dict], None],
        on_book_ticker:  Callable[[dict], None],
        on_depth:        Optional[Callable[[dict], None]] = None,
    ):
        """
        参数：
            symbols        : 要订阅的交易对列表（大小写均可，内部统一转小写）
            on_agg_trade   : 聚合成交回调，参数格式：
                             {"symbol": str, "price": float, "qty": float,
                              "is_buyer_maker": bool, "timestamp": int}
            on_book_ticker : 最优盘口回调，参数格式：
                             {"symbol": str, "bid": float, "bid_qty": float,
                              "ask": float, "ask_qty": float}
            on_depth       : Top-5 深度快照回调（可选），参数格式：
                             {"symbol": str, "bids": [[p,q],...], "asks": [[p,q],...],
                              "mid": float, "ts_ms": int}
                             来自 @depth5@100ms，每条消息是完整快照（非增量）
        """
        self.symbols        = [s.upper() for s in symbols]
        self.on_agg_trade   = on_agg_trade
        self.on_book_ticker = on_book_ticker
        self.on_depth       = on_depth

        self._running   = False
        self._threads:  List[threading.Thread]         = []
        self._ws_apps:  List[websocket.WebSocketApp]   = []
        self._ws_lock   = threading.Lock()
        self._proxy     = _parse_proxy()
        # depth5 时每连接 3 流/品种，60×3=180 < 200 仍安全
        self._streams_per_sym = 3 if on_depth else 2

        # 独立分发队列 + 消费线程：WS 线程只做入队，回调在此线程执行
        self._msg_queue: queue.Queue = queue.Queue(maxsize=DISPATCH_QUEUE_SIZE)
        self._dispatch_thread: Optional[threading.Thread] = None

    # ─── 公开接口 ────────────────────────────────────────────────────────────

    def start(self):
        """启动所有 WebSocket 连接（非阻塞，在后台线程运行）"""
        self._running = True

        # 先启动分发消费线程
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="MultiWS-Dispatch",
        )
        self._dispatch_thread.start()

        symbol_chunks = [
            self.symbols[i: i + SYMBOLS_PER_CONN]
            for i in range(0, len(self.symbols), SYMBOLS_PER_CONN)
        ]

        for i, chunk in enumerate(symbol_chunks):
            t = threading.Thread(
                target=self._run_connection,
                args=(chunk, i),
                daemon=True,
                name=f"MultiWS-{i}",
            )
            t.start()
            self._threads.append(t)
            time.sleep(0.5)   # 错开启动，避免同时连接触发限速

        logger.info(
            f"[MultiWS] 启动 {len(self._threads)} 个连接，"
            f"订阅 {len(self.symbols)} 个品种 × {self._streams_per_sym} 种流"
            + (" (含 depth5@100ms)" if self.on_depth else "")
        )

    def stop(self):
        """停止所有 WebSocket 连接"""
        self._running = False
        with self._ws_lock:
            apps = list(self._ws_apps)
        for ws in apps:
            try:
                ws.close()
            except Exception:
                pass
        # 唤醒分发线程使其退出
        try:
            self._msg_queue.put_nowait(None)
        except queue.Full:
            pass
        logger.info("[MultiWS] 所有连接已关闭")

    # ─── 内部实现 ────────────────────────────────────────────────────────────

    def _build_url(self, symbols: List[str]) -> str:
        """
        构建 Binance Combined Stream URL。

        例（含 depth）：
            btcusdt@aggTrade/btcusdt@bookTicker/btcusdt@depth5@100ms/...
        """
        streams = []
        for sym in symbols:
            sym_lower = sym.lower()
            streams.append(f"{sym_lower}@aggTrade")
            streams.append(f"{sym_lower}@bookTicker")
            if self.on_depth:
                streams.append(f"{sym_lower}@depth5@100ms")
        return f"{self.BASE_URL}?streams={'/'.join(streams)}"

    def _on_message(self, ws, raw: str):
        """
        WebSocket 消息处理器（仅解析 + 入队，不做业务回调）。

        业务回调在独立的 _dispatch_loop 线程中执行，
        避免回调耗时阻塞 WS recv 缓冲。

        Binance Combined Stream 的消息格式：
        {
            "stream": "btcusdt@aggTrade",
            "data": { ... }
        }
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        stream = msg.get("stream", "")
        data   = msg.get("data", {})
        if not stream or not data:
            return

        try:
            self._msg_queue.put_nowait((stream, data))
        except queue.Full:
            # 队列满时丢弃最旧的一条再入队（保持实时性，牺牲完整性）
            try:
                self._msg_queue.get_nowait()
                self._msg_queue.put_nowait((stream, data))
            except queue.Empty:
                pass

    def _dispatch_loop(self):
        """消息分发消费循环（独立线程，串行执行业务回调）。"""
        while self._running:
            try:
                item = self._msg_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:   # stop() 发出的退出信号
                break
            stream, data = item
            if "@aggTrade" in stream:
                self._handle_agg_trade(data)
            elif "@bookTicker" in stream:
                self._handle_book_ticker(data)
            elif "@depth" in stream and self.on_depth:
                self._handle_depth(data)

    def _handle_agg_trade(self, data: dict):
        """
        解析 aggTrade 数据。

        Binance aggTrade 字段：
            s : symbol
            p : price (str)
            q : quantity (str)
            m : is_buyer_maker (bool)  True=主动卖；False=主动买
            T : trade_time (ms)
        """
        try:
            self.on_agg_trade({
                "symbol":          data["s"],
                "price":           float(data["p"]),
                "qty":             float(data["q"]),
                "is_buyer_maker":  bool(data["m"]),
                "timestamp":       int(data["T"]),
            })
        except (KeyError, ValueError) as e:
            logger.debug(f"[MultiWS] aggTrade 解析错误: {e}")

    def _handle_book_ticker(self, data: dict):
        """
        解析 bookTicker 数据。

        Binance bookTicker 字段：
            s : symbol
            b : best bid price (str)
            B : best bid qty (str)
            a : best ask price (str)
            A : best ask qty (str)
        """
        try:
            self.on_book_ticker({
                "symbol":   data["s"],
                "bid":      float(data["b"]),
                "bid_qty":  float(data["B"]),
                "ask":      float(data["a"]),
                "ask_qty":  float(data["A"]),
            })
        except (KeyError, ValueError) as e:
            logger.debug(f"[MultiWS] bookTicker 解析错误: {e}")

    def _handle_depth(self, data: dict):
        """
        解析 @depth5@100ms 快照。

        每条消息是 top-5 bids/asks 的完整快照（非增量），无需维护本地订单簿状态。

        Binance depth5 字段：
            s : symbol
            b : [[price, qty], ...] bids（最多5档）
            a : [[price, qty], ...] asks（最多5档）
            T : transaction time (ms)
            E : event time (ms)
        """
        try:
            sym = data.get("s", "")
            if not sym:
                return
            bids_raw = data.get("b", [])
            asks_raw = data.get("a", [])
            bids = [[float(p), float(q)] for p, q in bids_raw if float(q) > 0]
            asks = [[float(p), float(q)] for p, q in asks_raw if float(q) > 0]
            if not bids or not asks:
                return
            mid = (bids[0][0] + asks[0][0]) / 2.0
            self.on_depth({
                "symbol": sym,
                "bids":   bids,
                "asks":   asks,
                "mid":    mid,
                "ts_ms":  int(data.get("T", data.get("E", 0))),
            })
        except (KeyError, ValueError, IndexError, TypeError) as e:
            logger.debug(f"[MultiWS] depth5 解析错误: {e}")

    def _on_error(self, ws, error):
        logger.error(f"[MultiWS] WebSocket 错误: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"[MultiWS] 连接关闭 code={close_status_code} msg={close_msg}")

    def _on_open(self, ws):
        logger.info("[MultiWS] WebSocket 连接已建立")

    def _run_connection(self, symbols: List[str], conn_idx: int):
        """
        运行单个 WebSocket 连接（含自动重连）。

        在独立线程中运行，断线后等待 RECONNECT_DELAY 秒重连。
        """
        url = self._build_url(symbols)
        logger.info(f"[MultiWS-{conn_idx}] 连接 {len(symbols)} 个品种 ({symbols[:3]}...)")

        while self._running:
            ws = websocket.WebSocketApp(
                url,
                on_message = self._on_message,
                on_error   = self._on_error,
                on_close   = self._on_close,
                on_open    = self._on_open,
            )
            with self._ws_lock:
                self._ws_apps.append(ws)
            ws.run_forever(
                ping_interval = 20,   # 每20s发一次ping，代理环境更频繁保活
                ping_timeout  = 10,   # 10s等不到pong就重连，快速失败
                **self._proxy,
            )
            with self._ws_lock:
                if ws in self._ws_apps:
                    self._ws_apps.remove(ws)

            if self._running:
                logger.info(f"[MultiWS-{conn_idx}] {RECONNECT_DELAY}s 后重连...")
                time.sleep(RECONNECT_DELAY)

"""
Multi-Symbol WebSocket Feed - 三通道分片并行版

架构：
    WS 线程（JSON 解析 + 分流）
        ├── trade_queues[N]   → N 个 trade 工作线程（按 symbol hash 分片）
        ├── _book_state[sym]  → book 消费线程（state-flow：覆盖最新值，无积压）
        └── _depth_state[sym] → depth 消费线程（state-flow：覆盖最新值，无积压）

核心改进：

    1. 三通道分离
       aggTrade / bookTicker / depth5 完全独立处理，互不阻塞

    2. 状态流优化（State-Flow）
       bookTicker / depth5：只保存最新值（覆盖旧值），消费线程按 5ms 轮询
       aggTrade：保持事件流（不覆盖），必须每条处理

    3. 分片并行（Symbol Sharding）
       worker_id = hash(symbol) % TRADE_WORKERS
       同 symbol 有序；不同 symbol 并行处理

    4. depth 动态订阅
       初始不订阅任何 depth5；调用 update_depth_symbols() 后动态增减
       只对「当前持仓 + TopN/BottomN + 候选池」开启，节省带宽

    5. 优先级控制
       trade > book > depth
       trade 队列满时丢弃最旧的一条；book/depth 状态流天然无积压

    6. 指数退避重连
       delay = min(BASE × 2^retry, MAX_DELAY=30s)
       连接稳定 60s 后重置重连计数器

    7. 实时监控统计
       每 60s 打印：各连接 msg/s、各 trade 队列 backlog、drop 数、reconnect 数

Binance Combined Stream 格式：
    wss://fstream.binance.com/stream?streams=btcusdt@aggTrade/btcusdt@bookTicker/...
    深度通过 SUBSCRIBE 消息动态添加：
    {"method": "SUBSCRIBE", "params": ["btcusdt@depth5@100ms"], "id": 1}
"""

import json
import math
import os
import queue
import time
import threading
import collections
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Set, Dict
from urllib.parse import urlparse

import websocket

from data_layer.logger import logger


# ── 常量 ──────────────────────────────────────────────────────────────────────

SYMBOLS_PER_CONN    = 40      # 每连接目标品种数（×2流=80，留 ~120 给 depth5 动态订阅）
TRADE_WORKERS       = 4       # trade 分片并行工作线程数（建议 ≈ CPU核数/2）
TRADE_QUEUE_SIZE    = 2_000   # 每个 trade shard 队列容量
RECONNECT_BASE      = 1.0     # 指数退避初始等待（秒）
RECONNECT_MAX       = 30.0    # 指数退避上限（秒）
RECONNECT_STABLE_S  = 60.0    # 连接稳定判断阈值：稳定超过此秒数后重置计数器
STATS_INTERVAL      = 60      # 监控统计打印间隔（秒）
STATE_POLL_INTERVAL = 0.005   # 状态流消费轮询间隔（5ms）


def _parse_proxy() -> dict:
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


@dataclass
class _ConnStats:
    """单个 WS 连接的实时统计计数器。"""
    msg_count:       int   = 0
    drop_count:      int   = 0
    reconnect_count: int   = 0
    is_connected:    bool  = False   # on_open 回调成功设为 True，断线设为 False
    _t0:             float = field(default_factory=time.time)

    def msg_per_sec(self) -> float:
        elapsed = time.time() - self._t0
        return self.msg_count / elapsed if elapsed > 0 else 0.0

    def reset_rate(self):
        """重置速率计数器（保留 drop/reconnect 累计值）。"""
        self.msg_count = 0
        self._t0 = time.time()


class MultiSymbolFeed:
    """
    Binance Futures Combined Stream 多币订阅客户端（三通道分片并行版）。

    外部接口：
        start()                  : 启动所有连接和工作线程
        stop()                   : 优雅关闭
        update_depth_symbols(s)  : 更新 depth5 动态订阅集合
        get_stats()              : 返回监控统计快照
    """

    BASE_URL = "wss://fstream.binance.com/stream"

    def __init__(
        self,
        symbols:        List[str],
        on_agg_trade:   Callable[[dict], None],
        on_book_ticker: Callable[[dict], None],
        on_depth:       Optional[Callable[[dict], None]] = None,
        trade_workers:  int = TRADE_WORKERS,
    ):
        """
        参数：
            symbols        : 交易对列表（大小写均可，内部转大写）
            on_agg_trade   : aggTrade 回调，格式：
                             {"symbol", "price", "qty", "is_buyer_maker", "timestamp"}
            on_book_ticker : bookTicker 回调，格式：
                             {"symbol", "bid", "bid_qty", "ask", "ask_qty"}
            on_depth       : depth5 回调（可选），格式：
                             {"symbol", "bids", "asks", "mid", "ts_ms"}
            trade_workers  : trade 通道分片数（并行工作线程数）
        """
        self.symbols        = [s.upper() for s in symbols]
        self.on_agg_trade   = on_agg_trade
        self.on_book_ticker = on_book_ticker
        self.on_depth       = on_depth
        self._running       = False
        self._proxy         = _parse_proxy()

        # ── 1. trade 通道：N 个分片队列 ──────────────────────────────────────
        self._n_workers = trade_workers
        self._trade_queues: List[queue.Queue] = [
            queue.Queue(maxsize=TRADE_QUEUE_SIZE) for _ in range(self._n_workers)
        ]
        # 预计算 symbol → shard 映射（O(1) 路由）
        self._sym_to_shard: Dict[str, int] = {
            sym: hash(sym) % self._n_workers for sym in self.symbols
        }

        # ── 2. book 通道：状态流（latest-only）──────────────────────────────
        self._book_state: Dict[str, dict] = {}
        self._book_dirty: Set[str]        = set()
        self._book_lock   = threading.Lock()

        # ── 3. depth 通道：状态流（latest-only）─────────────────────────────
        self._depth_state: Dict[str, dict] = {}
        self._depth_dirty: Set[str]        = set()
        self._depth_lock   = threading.Lock()

        # ── 4. depth 动态订阅状态 ─────────────────────────────────────────────
        self._depth_active:   Set[str]                    = set()
        self._depth_lock_sub  = threading.Lock()
        self._sym_to_conn:    Dict[str, int]             = {}   # symbol → 连接 index
        self._conn_ws:        Dict[int, websocket.WebSocketApp] = {}
        self._ws_lock         = threading.Lock()
        self._sub_req_id      = 0

        # ── 5. 监控统计 ───────────────────────────────────────────────────────
        self._conn_stats: Dict[int, _ConnStats] = collections.defaultdict(_ConnStats)
        self._total_drop              = 0
        self._sf_book_updates         = 0   # state-flow book 更新次数（60s 窗口）
        self._sf_depth_updates        = 0   # state-flow depth 更新次数（60s 窗口）

        # ── 后台线程列表 ──────────────────────────────────────────────────────
        self._threads: List[threading.Thread] = []

    # ─── 公开接口 ────────────────────────────────────────────────────────────

    def start(self):
        """启动所有工作线程和 WS 连接（非阻塞）。"""
        self._running = True

        # 1. N 个 trade 分片工作线程
        for i in range(self._n_workers):
            self._start_thread(self._trade_worker, (i,), f"TradeWorker-{i}")

        # 2. book 状态流消费线程
        self._start_thread(self._book_consumer, (), "BookConsumer")

        # 3. depth 状态流消费线程（仅在注册了 on_depth 时启动）
        if self.on_depth:
            self._start_thread(self._depth_consumer, (), "DepthConsumer")

        # 4. 监控统计线程
        self._start_thread(self._stats_loop, (), "WSStats")

        # 5. WS 连接线程（round-robin 均匀打散，而不是连续切块）
        symbol_chunks = self._build_round_robin_chunks(self.symbols, SYMBOLS_PER_CONN)

        if not symbol_chunks:
            logger.warning("[MultiWS] symbols 列表为空，未启动任何 WS 连接！请检查交易池过滤结果。")

        # 重建 symbol -> conn 映射
        self._sym_to_conn.clear()
        for conn_idx, chunk in enumerate(symbol_chunks):
            for sym in chunk:
                self._sym_to_conn[sym] = conn_idx

        # 启动各连接
        for conn_idx, chunk in enumerate(symbol_chunks):
            _ = self._conn_stats[conn_idx]  # 预注册 stats
            self._start_thread(self._run_connection, (chunk, conn_idx), f"MultiWS-{conn_idx}")
            time.sleep(0.5)   # 错开启动，避免同时握手触发限速

        chunk_sizes = [len(chunk) for chunk in symbol_chunks]
        logger.info(
            f"[MultiWS] 启动 | "
            f"品种={len(self.symbols)} | "
            f"连接数={len(symbol_chunks)} | "
            f"每连接品种数={chunk_sizes} | "
            f"trade分片={self._n_workers} | "
            f"depth=动态（初始 0） | "
            f"symbol分配=round-robin"
        )

    def stop(self):
        """优雅关闭所有连接和线程。"""
        self._running = False

        with self._ws_lock:
            for ws in self._conn_ws.values():
                try:
                    ws.close()
                except Exception:
                    pass

        # 向所有 trade 队列发送退出哨兵
        for q in self._trade_queues:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

        logger.info("[MultiWS] 已关闭")

    def update_depth_symbols(self, wanted: Set[str]):
        """
        更新 depth5 动态订阅集合（差量更新）。

        调用方：AlphaFactoryStrategy 每轮 _rank_and_trade() 结束后调用：
            ws_feed.update_depth_symbols(
                set(long_positions.keys()) |
                set(short_positions.keys()) |
                top_long_candidates |
                top_short_candidates
            )

        参数：
            wanted : 需要 depth5 的品种集合（大写），传入空集合则全部取消订阅
        """
        if not self.on_depth:
            return

        wanted_upper = {s.upper() for s in wanted}

        with self._depth_lock_sub:
            to_add    = wanted_upper - self._depth_active
            to_remove = self._depth_active - wanted_upper

            for sym in to_add:
                self._send_sub(sym, subscribe=True)

            for sym in to_remove:
                self._send_sub(sym, subscribe=False)

            if to_add or to_remove:
                logger.debug(
                    f"[MultiWS] depth +{len(to_add)} -{len(to_remove)} "
                    f"→ 共 {len(wanted_upper)} 个"
                )
            self._depth_active = set(wanted_upper)

    def get_stats(self) -> dict:
        """返回当前监控快照，供外部仪表盘查询。"""
        with self._depth_lock_sub:
            n_depth = len(self._depth_active)
        return {
            "connections": {
                idx: {
                    "msg_per_sec":     round(s.msg_per_sec(), 2),
                    "drop_count":      s.drop_count,
                    "reconnect_count": s.reconnect_count,
                }
                for idx, s in self._conn_stats.items()
            },
            "trade_queue_backlog": [q.qsize() for q in self._trade_queues],
            "total_drop":          self._total_drop,
            "depth_active":        n_depth,
        }

    # ─── 内部工具 ────────────────────────────────────────────────────────────

    def _start_thread(self, target, args, name):
        t = threading.Thread(target=target, args=args, daemon=True, name=name)
        t.start()
        self._threads.append(t)

    def _build_round_robin_chunks(self, symbols: List[str], symbols_per_conn: int) -> List[List[str]]:
        """
        按 round-robin 将 symbol 均匀打散到各连接。

        目标：
            - 保持总连接数 ≈ ceil(len(symbols) / symbols_per_conn)
            - 不再使用连续切块，避免“前段 symbol 全落到 conn-0”
            - 让 depth 动态订阅也随 _sym_to_conn 自然均摊

        例子：
            symbols=[s1,s2,s3,s4,s5,s6], 每连接目标=2
            连接数=3
            结果：
                conn-0: s1,s4
                conn-1: s2,s5
                conn-2: s3,s6
        """
        if not symbols:
            return []

        n_conn = max(1, math.ceil(len(symbols) / symbols_per_conn))
        chunks: List[List[str]] = [[] for _ in range(n_conn)]

        for i, sym in enumerate(symbols):
            conn_idx = i % n_conn
            chunks[conn_idx].append(sym)

        return [chunk for chunk in chunks if chunk]

    def _send_sub(self, sym: str, subscribe: bool):
        """向对应 WS 连接发送 SUBSCRIBE / UNSUBSCRIBE 消息（已持有 _depth_lock_sub）。"""
        conn_idx = self._sym_to_conn.get(sym)
        if conn_idx is None:
            return
        with self._ws_lock:
            ws = self._conn_ws.get(conn_idx)
        if ws is None:
            return
        self._sub_req_id += 1
        method = "SUBSCRIBE" if subscribe else "UNSUBSCRIBE"
        try:
            ws.send(json.dumps({
                "method": method,
                "params": [f"{sym.lower()}@depth5@100ms"],
                "id":     self._sub_req_id,
            }))
        except Exception as e:
            logger.debug(f"[MultiWS] {method} {sym} 失败: {e}")

    # ─── WS 连接（指数退避重连）──────────────────────────────────────────────

    def _build_url(self, symbols: List[str]) -> str:
        """构建只含 aggTrade + bookTicker 的 Combined Stream URL。"""
        streams = []
        for sym in symbols:
            s = sym.lower()
            streams.append(f"{s}@aggTrade")
            streams.append(f"{s}@bookTicker")
        return f"{self.BASE_URL}?streams={'/'.join(streams)}"

    def _run_connection(self, symbols: List[str], conn_idx: int):
        """
        单个 WS 连接主循环（含指数退避重连）。

        重连策略：
            delay = min(RECONNECT_BASE × 2^retry, RECONNECT_MAX)
            连接稳定 ≥ RECONNECT_STABLE_S 秒后，retry 归零
        """
        url   = self._build_url(symbols)
        stats = self._conn_stats[conn_idx]
        retry = 0

        logger.info(f"[MultiWS-{conn_idx}] 启动 | {len(symbols)} 个品种 ({symbols[:3]}...)")

        while self._running:
            connect_start = time.time()

            ws = websocket.WebSocketApp(
                url,
                on_message=lambda _, m: self._on_message(m, conn_idx),
                on_error=lambda _, e: self._on_error(e, conn_idx),
                on_close=lambda _, c, m: self._on_close(c, m, conn_idx),
                on_open=lambda _w: self._on_open(_w, conn_idx),
            )
            ws.run_forever(
                ping_interval=30,   # 每 30s 发一次 ping（必须 > ping_timeout）
                ping_timeout=10,    # 等 pong 最多 10s（interval > timeout，满足库约束）
                **self._proxy,
            )

            # 连接断开，清理 ws 引用
            with self._ws_lock:
                if self._conn_ws.get(conn_idx) is ws:
                    del self._conn_ws[conn_idx]

            if not self._running:
                break

            # 指数退避：连接稳定一段时间则重置
            stable = time.time() - connect_start >= RECONNECT_STABLE_S
            if stable:
                retry = 0
            delay = min(RECONNECT_BASE * (2 ** retry), RECONNECT_MAX)
            retry += 1
            stats.reconnect_count += 1

            logger.warning(
                f"[MultiWS-{conn_idx}] 断线 "
                f"(第 {stats.reconnect_count} 次重连) "
                f"→ {delay:.1f}s 后重试"
            )
            time.sleep(delay)

    def _on_open(self, ws, conn_idx: int):
        """连接建立：注册 ws 引用，恢复已激活的 depth 订阅。"""
        with self._ws_lock:
            self._conn_ws[conn_idx] = ws
        self._conn_stats[conn_idx].is_connected = True

        # 恢复该连接负责的 depth 订阅（断线重连后需重新订阅）
        with self._depth_lock_sub:
            to_restore = {
                sym for sym in self._depth_active
                if self._sym_to_conn.get(sym) == conn_idx
            }
        for sym in to_restore:
            with self._depth_lock_sub:
                self._send_sub(sym, subscribe=True)

        logger.info(
            f"[MultiWS-{conn_idx}] 已连接"
            + (f"，恢复 {len(to_restore)} 个 depth 订阅" if to_restore else "")
        )

    def _on_error(self, error, conn_idx: int):
        logger.error(f"[MultiWS-{conn_idx}] 错误: {error}")

    def _on_close(self, close_status_code, close_msg, conn_idx: int):
        self._conn_stats[conn_idx].is_connected = False
        logger.warning(
            f"[MultiWS-{conn_idx}] 关闭 "
            f"code={close_status_code} msg={close_msg}"
        )

    # ─── 消息分流（WS 线程，极低延迟）──────────────────────────────────────────

    def _on_message(self, raw: str, conn_idx: int):
        """
        WS 消息接收器（运行在 WS 线程）。

        只做：JSON 解析 + 分流到三通道
        不做任何业务计算，确保 WS recv 缓冲不积压。

        Binance Combined Stream 消息格式：
            {"stream": "btcusdt@aggTrade", "data": {...}}
        """
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        stream = msg.get("stream", "")
        data   = msg.get("data")
        if not stream or data is None:
            return

        stats = self._conn_stats[conn_idx]
        stats.msg_count += 1

        if "@aggTrade" in stream:
            self._route_trade(data, stats)
        elif "@bookTicker" in stream:
            self._route_book(data)
        elif "@depth" in stream:
            self._route_depth(data)

    def _route_trade(self, data: dict, stats: _ConnStats):
        """
        trade 消息 → 按 symbol hash 分片入队（事件流，不覆盖）。

        优先级最高：队列满时丢最旧的一条，而不是丢当前消息。
        """
        sym = data.get("s", "")
        if not sym:
            return
        shard = self._sym_to_shard.get(sym, hash(sym) % self._n_workers)
        q = self._trade_queues[shard]
        try:
            q.put_nowait(data)
        except queue.Full:
            # 丢最旧 → 保证实时性
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(data)
            except queue.Full:
                pass
            stats.drop_count += 1
            self._total_drop += 1

    def _route_book(self, data: dict):
        """
        book 消息 → 覆盖最新值（state-flow）。

        不入队，直接覆盖 _book_state[sym]，消费线程轮询 dirty 集合处理。
        天然无积压：不论推送多快，消费侧始终看到最新值。
        """
        sym = data.get("s", "")
        if not sym:
            return
        with self._book_lock:
            self._book_state[sym] = data
            self._book_dirty.add(sym)
        self._sf_book_updates += 1

    def _route_depth(self, data: dict):
        """
        depth 消息 → 覆盖最新值（state-flow）。

        优先级最低：state-flow 天然无积压，无需额外丢弃逻辑。
        """
        sym = data.get("s", "")
        if not sym:
            return
        with self._depth_lock:
            self._depth_state[sym] = data
            self._depth_dirty.add(sym)
        self._sf_depth_updates += 1

    # ─── 工作线程 ────────────────────────────────────────────────────────────

    def _trade_worker(self, shard_id: int):
        """
        trade 分片工作线程。

        从对应 shard 队列消费，调用 on_agg_trade 回调。
        同 symbol 的消息在同一 shard，保证 symbol 级有序。
        不同 shard 的 symbol 并行处理。
        """
        q = self._trade_queues[shard_id]
        while self._running:
            try:
                data = q.get(timeout=1.0)
            except queue.Empty:
                continue
            if data is None:   # 退出哨兵
                break
            try:
                self.on_agg_trade({
                    "symbol":         data["s"],
                    "price":          float(data["p"]),
                    "qty":            float(data["q"]),
                    "is_buyer_maker": bool(data["m"]),
                    "timestamp":      int(data["T"]),
                })
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"[TradeWorker-{shard_id}] 解析错误: {e}")

    def _book_consumer(self):
        """
        book 状态流消费线程。

        每 5ms 轮询 dirty 集合，对每个有更新的 symbol 调用 on_book_ticker。
        拍摄快照后立即清空 dirty，避免持锁时间过长。
        """
        while self._running:
            time.sleep(STATE_POLL_INTERVAL)
            with self._book_lock:
                if not self._book_dirty:
                    continue
                dirty    = list(self._book_dirty)
                self._book_dirty.clear()
                snapshot = {sym: self._book_state[sym] for sym in dirty if sym in self._book_state}

            for sym, data in snapshot.items():
                try:
                    self.on_book_ticker({
                        "symbol":  sym,
                        "bid":     float(data["b"]),
                        "bid_qty": float(data["B"]),
                        "ask":     float(data["a"]),
                        "ask_qty": float(data["A"]),
                    })
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(f"[BookConsumer] {sym} 解析错误: {e}")

    def _depth_consumer(self):
        """
        depth 状态流消费线程。

        每 5ms 轮询 dirty 集合，对每个有更新的 symbol 调用 on_depth。
        """
        while self._running:
            time.sleep(STATE_POLL_INTERVAL)
            with self._depth_lock:
                if not self._depth_dirty:
                    continue
                dirty    = list(self._depth_dirty)
                self._depth_dirty.clear()
                snapshot = {sym: self._depth_state[sym] for sym in dirty if sym in self._depth_state}

            for sym, data in snapshot.items():
                try:
                    bids_raw = data.get("b", [])
                    asks_raw = data.get("a", [])
                    bids = [[float(p), float(q)] for p, q in bids_raw if float(q) > 0]
                    asks = [[float(p), float(q)] for p, q in asks_raw if float(q) > 0]
                    if not bids or not asks:
                        continue
                    mid = (bids[0][0] + asks[0][0]) / 2.0
                    self.on_depth({
                        "symbol": sym,
                        "bids":   bids,
                        "asks":   asks,
                        "mid":    mid,
                        "ts_ms":  int(data.get("T", data.get("E", 0))),
                    })
                except (KeyError, ValueError, IndexError, TypeError) as e:
                    logger.debug(f"[DepthConsumer] {sym} 解析错误: {e}")

    # ─── 监控统计 ─────────────────────────────────────────────────────────────

    def _stats_loop(self):
        """监控统计打印线程（每 60s 打印一次）。"""
        while self._running:
            time.sleep(STATS_INTERVAL)
            if not self._running:
                break
            self._print_stats()

    def _print_stats(self):
        sep = "-" * 62
        logger.info(sep)
        logger.info(f"[MultiWS Stats] {time.strftime('%H:%M:%S')}")

        # 各连接 msg/s、drop、reconnect、连接状态
        for conn_idx, stats in sorted(self._conn_stats.items()):
            status = "✓ UP" if stats.is_connected else "✗ DOWN"
            logger.info(
                f"  conn-{conn_idx}:  "
                f"[{status}]  "
                f"msg/s={stats.msg_per_sec():>7.1f}  "
                f"drops={stats.drop_count:>5}  "
                f"reconnects={stats.reconnect_count}"
            )
            stats.reset_rate()

        # trade 队列积压
        backlogs = [q.qsize() for q in self._trade_queues]
        logger.info(
            f"  trade_q backlog : {backlogs}  "
            f"total_drop={self._total_drop}"
        )

        # 状态流更新速率（60s 窗口内）
        with self._depth_lock_sub:
            n_depth = len(self._depth_active)
        logger.info(
            f"  state-flow/60s  : book={self._sf_book_updates}  "
            f"depth={self._sf_depth_updates}"
        )
        logger.info(f"  depth 订阅数    : {n_depth} 个")

        self._sf_book_updates  = 0
        self._sf_depth_updates = 0
        logger.info(sep)
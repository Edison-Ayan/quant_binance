"""
交易数据存储模块（Database）

职责：
    将系统内每一笔成交记录（FILL 事件）持久化到本地 SQLite 数据库，
    用于事后分析、绩效统计和审计追踪。

在系统事件流中的位置：
    FILL 事件到达（BinanceGateway 发出）
        │  Database.on_fill_event()
        │  → 解析成交数据
        │  → 调用 save_trade() 写入 SQLite
        ▼
    本地 SQLite 文件（quant_trades.db）

设计要点：
    1. 线程安全：EventEngine 在独立线程中调用 handler，SQLite 写操作需加锁
       使用 threading.Lock 保护 conn.execute + conn.commit 的原子性
    2. check_same_thread=False：允许在非创建线程中使用同一连接，
       配合 threading.Lock 确保同一时刻只有一个线程在操作连接
    3. 时间戳精度：使用毫秒级 UNIX 时间戳（int(time.time() * 1000)），
       与 Binance API 的时间精度保持一致，方便对账

数据库 Schema（trades 表）：
    id        : 自增主键，无业务意义
    order_id  : 交易所订单 ID（Binance orderId）
    symbol    : 交易对，例如 "BTCUSDT"
    side      : 方向，"BUY" 或 "SELL"
    price     : 成交均价
    qty       : 本次成交数量
    timestamp : 成交时间（毫秒级 UNIX 时间戳）
    status    : 订单状态（FILLED / PARTIALLY_FILLED 等）
"""

import sqlite3
import threading
import time
import queue as _queue


class Database:
    """
    基于 SQLite 的本地成交记录数据库。

    通过 FILL 事件监听自动写入，并提供成交记录查询接口。
    """

    def __init__(self, db_path: str):
        """
        初始化数据库连接，并确保 trades 表存在。

        参数：
            db_path (str) : SQLite 数据库文件路径，例如 "quant_trades.db"
                            文件不存在时 SQLite 会自动创建
        """
        self.db_path = db_path

        # 线程锁：保护 SQLite 写操作的原子性
        # 虽然 SQLite 有自己的文件级锁，但 threading.Lock 可以在进程内
        # 避免多线程争抢连接导致的 "database is locked" 错误
        self._lock = threading.Lock()

        # 创建 SQLite 连接
        # check_same_thread=False：允许在非创建线程中使用此连接
        # EventEngine 的消费线程与主线程不同，必须设置此参数
        # 配合 threading.Lock 保证线程安全
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        # 初始化数据表（幂等，若表已存在则不重建）
        self._create_table()

        # 异步写入队列：on_fill_event 只入队（O(1) 非阻塞），
        # 后台线程负责 SQLite commit，不占用 EventEngine 线程
        self._write_queue: _queue.Queue = _queue.Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="db-writer"
        )
        self._writer_thread.start()

    # -------------------------
    # 数据库初始化
    # -------------------------

    def _create_table(self):
        """创建数据表（幂等，若表已存在则不重建）。"""
        with self._lock:
            # 原始订单表（每笔开/平单独一行）
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id   TEXT,
                    symbol     TEXT    NOT NULL,
                    side       TEXT    NOT NULL,
                    price      REAL    NOT NULL,
                    qty        REAL    NOT NULL,
                    timestamp  INTEGER NOT NULL,
                    status     TEXT
                )
                """
            )
            # 完整交易周期表（开仓→平仓 合并为一行，含盈亏）
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS completed_trades (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol        TEXT    NOT NULL,
                    side          TEXT    NOT NULL,
                    entry_price   REAL    NOT NULL,
                    exit_price    REAL    NOT NULL,
                    qty           REAL    NOT NULL,
                    leverage      INTEGER NOT NULL,
                    pnl_usdt      REAL    NOT NULL,
                    ret_pct       REAL    NOT NULL,
                    ret_lev_pct   REAL    NOT NULL,
                    hold_seconds  REAL    NOT NULL,
                    reason        TEXT,
                    entry_time    INTEGER NOT NULL,
                    exit_time     INTEGER NOT NULL
                )
                """
            )
            self.conn.commit()

    # -------------------------
    # 事件处理
    # -------------------------

    def _writer_loop(self):
        """
        后台写入线程：持续从队列取记录并写入 SQLite。
        队列元素格式：(table_name, values_tuple)
        None 作为哨兵值触发退出。
        """
        SQL = {
            "trades": (
                "INSERT INTO trades "
                "(order_id, symbol, side, price, qty, timestamp, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            "completed_trades": (
                "INSERT INTO completed_trades "
                "(symbol, side, entry_price, exit_price, qty, leverage, "
                " pnl_usdt, ret_pct, ret_lev_pct, hold_seconds, reason, "
                " entry_time, exit_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
        }
        while True:
            item = self._write_queue.get()
            if item is None:
                break
            table, values = item
            with self._lock:
                try:
                    self.conn.execute(SQL[table], values)
                    self.conn.commit()
                except Exception as e:
                    print(f"[Database] write error ({table}): {e}")

    def on_fill_event(self, event):
        """接收 FILL 事件，非阻塞入队写入 trades 表。"""
        data = event.data
        self._write_queue.put(("trades", (
            str(data.get("order_id", "")),
            data.get("symbol", ""),
            data.get("side", ""),
            float(data.get("price", 0)),
            float(data.get("last_qty", 0)),
            int(time.time() * 1000),
            data.get("status", ""),
        )))

    # -------------------------
    # 数据写入
    # -------------------------

    def save_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        order_id: str = "",
        status: str = "",
    ):
        """将一条原始订单记录异步写入 trades 表（非阻塞）。"""
        self._write_queue.put(("trades", (
            order_id, symbol, side, price, qty, int(time.time() * 1000), status,
        )))

    def save_completed_trade(self, trade) -> None:
        """
        将一个完整开→平交易周期写入 completed_trades 表（非阻塞）。

        参数：
            trade : alpha_strategy.TradeRecord 实例
        """
        self._write_queue.put(("completed_trades", (
            trade.symbol,
            trade.side,
            trade.entry_price,
            trade.exit_price,
            trade.qty,
            trade.leverage,
            trade.pnl_usdt,
            trade.ret_pct,
            trade.ret_lev_pct,
            trade.hold_seconds,
            trade.reason,
            int(trade.entry_time * 1000),
            int(trade.exit_time  * 1000),
        )))

    def close(self):
        """停止后台写入线程，刷完剩余队列后关闭数据库连接。"""
        self._write_queue.put(None)     # 发送哨兵，让 _writer_loop 退出
        self._writer_thread.join(timeout=5)

    # -------------------------
    # 数据查询
    # -------------------------

    def get_completed_trades(self, symbol: str = None, limit: int = 200) -> list:
        """
        查询完整交易周期记录（completed_trades 表），按时间降序。

        返回字段顺序：
            (id, symbol, side, entry_price, exit_price, qty, leverage,
             pnl_usdt, ret_pct, ret_lev_pct, hold_seconds, reason,
             entry_time, exit_time)
        """
        with self._lock:
            if symbol:
                cursor = self.conn.execute(
                    "SELECT * FROM completed_trades WHERE symbol=? "
                    "ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor = self.conn.execute(
                    "SELECT * FROM completed_trades ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            return cursor.fetchall()

    def get_trades(self, symbol: str = None, limit: int = 100) -> list:
        """
        查询成交记录，按时间降序（最新的在前），支持按品种过滤。

        参数：
            symbol (str)  : 可选，按交易对过滤；None 表示查询所有品种
            limit  (int)  : 最多返回的记录数，默认 100，防止一次性加载过多数据

        返回：
            list : 成交记录列表，每条记录为 tuple，字段顺序：
                   (id, order_id, symbol, side, price, qty, timestamp, status)
        """
        with self._lock:
            if symbol:
                # 按品种过滤：使用参数化查询防止注入
                cursor = self.conn.execute(
                    "SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                # 查询所有品种
                cursor = self.conn.execute(
                    "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            return cursor.fetchall()

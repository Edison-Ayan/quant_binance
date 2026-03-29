"""
Binance WebSocket 行情客户端

职责：
    通过 Binance 合约逐笔成交流（@trade）实时接收最新成交价格，
    并将每笔数据封装为 TICK 事件投递到 EventEngine，供策略和持仓模块消费。

连接地址：
    wss://fstream.binance.com/ws/{symbol}@trade
    （fstream 为合约端点；现货行情使用 stream.binance.com）

数据格式（Binance @trade 推送字段）：
    {
        "e": "trade",   # 事件类型
        "E": 1234567,   # 事件时间（ms）
        "s": "BTCUSDT", # 交易对
        "t": 12345,     # 成交ID
        "p": "45000.0", # 成交价格（字符串）
        "q": "0.001",   # 成交数量
        "T": 1234567,   # 成交时间
        "m": true       # 是否为做市方
    }

在系统事件流中的位置：
    BinanceWebSocketClient (本模块)
        │  发出 TICK{symbol, price}
        ▼
    EventEngine → PositionManager.on_tick / MACrossStrategy.on_tick
"""

import json
import time
import websocket

from core.event import Event
from core.constants import EventType
from data_layer.logger import logger


class BinanceWebSocketClient:
    """
    Binance 合约逐笔行情 WebSocket 客户端。

    每个实例订阅一个交易对的实时成交流，将成交价包装为 TICK 事件
    持续投递到 EventEngine，驱动后续策略计算和持仓浮盈更新。
    """

    def __init__(self, symbol: str, event_engine):
        """
        初始化客户端。

        参数：
            symbol       (str)         : 交易对，例如 "BTCUSDT"（大小写均可）
            event_engine (EventEngine) : 全局事件引擎，用于投递 TICK 事件
        """
        # 转小写以拼接 WebSocket URL（Binance 要求 URL 中的 symbol 为小写）
        self.symbol = symbol.lower()
        self.event_engine = event_engine
        self._running = True

    def on_message(self, ws, message):
        """
        WebSocket 消息回调（每收到一笔成交数据触发一次）。

        解析 JSON 消息，提取成交价格字段 "p"，封装为 TICK 事件
        投递到 EventEngine。TICK 事件的 data 格式：
            {"symbol": "BTCUSDT", "price": 45000.0}

        参数：
            ws      : websocket.WebSocketApp 实例（框架传入，通常不使用）
            message (str) : 从服务端收到的原始 JSON 字符串
        """
        # 将 JSON 字符串解析为字典
        data = json.loads(message)

        # p：成交价格（price），字符串格式，需转 float
        price = float(data["p"])

        # q：成交数量（quantity），字符串格式，需转 float
        qty = float(data["q"])

        # m：是否为做市方（is_buyer_maker）
        # True  = 买方为做市方 → 该笔成交由卖方主动发起（主动卖单）→ 价格下行压力
        # False = 卖方为做市方 → 该笔成交由买方主动发起（主动买单）→ 价格上行压力
        # 用于 Hawkes 过程中区分主动买/卖强度
        is_buyer_maker = data["m"]

        # T：成交时间戳（毫秒），用于 Hawkes 过程的时间衰减计算
        timestamp = data["T"]

        # 封装完整的 TICK 事件：携带价格、数量、方向、时间戳
        # 高频策略（Hawkes 过程）需要 qty / is_buyer_maker / timestamp
        event = Event(
            EventType.TICK,
            {
                "symbol":         self.symbol.upper(),  # 统一大写，例如 "BTCUSDT"
                "price":          price,                # 最新成交价
                "qty":            qty,                  # 成交数量
                "is_buyer_maker": is_buyer_maker,       # 主动方向（False=主动买）
                "timestamp":      timestamp             # 成交时间戳（ms）
            }
        )

        # 将 TICK 事件投入事件引擎队列，由消费线程异步分发给注册的 handler
        self.event_engine.put(event)

    def _on_error(self, ws, error):
        logger.error(f"[TICK:{self.symbol.upper()}] WebSocket error: {error}")

    def _on_close(self, ws, code, msg):
        logger.warning(f"[TICK:{self.symbol.upper()}] WebSocket closed code={code}")

    def stop(self):
        """停止重连循环（主程序退出时调用）。"""
        self._running = False

    def start(self):
        """
        建立 WebSocket 连接并阻塞监听；断线后自动重连（通常在独立线程中调用）。
        """
        url = f"wss://fstream.binance.com/ws/{self.symbol}@trade"
        while self._running:
            ws = websocket.WebSocketApp(
                url,
                on_message=self.on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=20)
            if self._running:
                logger.info(f"[TICK:{self.symbol.upper()}] 3s 后重连...")
                time.sleep(3)

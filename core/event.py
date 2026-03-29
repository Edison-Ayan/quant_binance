"""
事件对象定义

职责：
    Event 是系统内所有模块间传递信息的唯一载体。
    生产者（策略、网关、行情客户端）创建 Event 并调用 EventEngine.put() 放入队列，
    消费者（风控、持仓、执行层）由 EventEngine 调用其 handler 函数并传入 Event。

设计原则：
    - 轻量：只包含类型标签（type）和数据字典（data），无任何业务逻辑
    - 不可变式：使用 dataclass 定义，字段明确，防止意外添加属性
    - 通用：data 字段为 dict，各事件类型自定义所需字段，无需为每种事件
            单独建类，降低了系统复杂度

常见 data 字段约定（各生产者遵循此约定，消费者依赖此约定解析）：
    TICK    : {"symbol": str, "price": float, "qty": float,
               "is_buyer_maker": bool, "timestamp": int}
              ── 由 BinanceWebSocketClient 发出，携带逐笔成交完整信息
              ── is_buyer_maker=True 表示主动卖单（做市方为买方），即下跌成交
              ── is_buyer_maker=False 表示主动买单（做市方为卖方），即上涨成交

    BAR     : {"symbol": str, "open": float, "high": float, "low": float,
               "close": float, "volume": float, "timestamp": int}
              ── 由K线合成模块发出，timestamp 为该 Bar 开始时间（ms）

    ORDER_BOOK : {"symbol": str, "timestamp": int,
                  "bids": [[price, qty], ...],  # 买盘 top-N 档，价格降序
                  "asks": [[price, qty], ...],  # 卖盘 top-N 档，价格升序
                  "best_bid": float, "best_ask": float,
                  "best_bid_qty": float, "best_ask_qty": float}
              ── 由 OrderBookEngine 在每次增量更新后发出，供高频策略计算 OFI 等指标

    SIGNAL  : {"symbol": str, "side": str, "qty": float, "price": float}
              ── 由策略发出，side 为 "BUY" 或 "SELL"

    ORDER   : {"symbol": str, "side": str, "qty": float, "price": float}
              ── 由 RiskManager 风控通过后发出，OrderManager 负责执行

    FILL    : {"symbol": str, "side": str, "order_id": int,
               "status": str, "last_qty": float, "cum_qty": float, "price": float}
              ── 由 BinanceGateway 从交易所回报解析后发出

    ACCOUNT : {"balances": list, "positions": list}
              ── 由 BinanceGateway 从 ACCOUNT_UPDATE 推送解析后发出
"""

from dataclasses import dataclass
from core.constants import EventType


@dataclass
class Event:
    """
    系统事件对象

    所有模块间通信的统一数据结构。EventEngine 根据 type 字段将事件路由
    到对应的 handler 列表，handler 再从 data 字段中提取所需的业务数据。

    属性：
        type (EventType) : 事件类型，决定由哪些 handler 处理（EventEngine 路由键）
        data (dict)      : 事件携带的业务数据，key-value 格式，字段含义见模块文档约定
    """

    type: EventType   # 事件类型（用于 EventEngine 路由到对应 handler）
    data: dict        # 业务数据（key-value 格式，字段含义见模块文档）

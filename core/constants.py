"""
系统常量定义

职责：
    集中定义系统内所有模块共用的枚举类型，避免魔法字符串散落各处，
    便于统一维护、重构和 IDE 自动补全。

设计原则：
    - 所有模块只从此处 import 常量，不在业务代码中硬编码字符串
    - 枚举值与字符串保持一致（EventType.TICK.value == "TICK"），
      方便调试时直接打印可读的名称
"""

from enum import Enum


class EventType(Enum):
    """
    事件类型枚举

    系统内所有模块通过 EventEngine 的事件队列解耦通信，
    每种事件有固定的生产者（emit）和消费者（handler）：

    ┌────────────┬──────────────────────────────┬─────────────────────────────────────────┐
    │ 事件       │ 生产者                        │ 消费者                                   │
    ├────────────┼──────────────────────────────┼─────────────────────────────────────────┤
    │ TICK       │ BinanceWebSocketClient        │ PositionManager, Strategy.on_tick        │
    │ MARKET     │ （兼容保留，不再主动使用）     │ -                                        │
    │ BAR        │ K线合成模块                   │ Strategy.on_bar                          │
    │ ORDER_BOOK │ OrderBookEngine               │ 高频策略（OFI / Hawkes / 盘口压力）      │
    │ SIGNAL     │ Strategy（均线/信号触发）      │ RiskManager（风控校验）                  │
    │ ORDER      │ RiskManager（风控通过后发出）  │ OrderManager（调用网关下单）             │
    │ FILL       │ BinanceGateway（成交回报）     │ PositionManager, RiskManager, DB         │
    │ ACCOUNT    │ BinanceGateway（余额变动）     │ AccountManager（同步余额）               │
    └────────────┴──────────────────────────────┴─────────────────────────────────────────┘
    """

    # 逐笔行情：BinanceWebSocketClient 通过 @trade 流推送，包含最新成交价
    # 消费者：PositionManager（更新浮盈）、Strategy.on_tick
    TICK    = "TICK"

    # K线数据（OHLCV）：供策略做周期性分析，目前为占位符，未来由K线合成模块发出
    BAR     = "BAR"

    # 策略交易信号：包含方向（BUY/SELL）、数量、参考价格，由策略逻辑判断后发出
    # 注意：信号不等于订单，必须经过 RiskManager 风控校验才能变成 ORDER
    SIGNAL  = "SIGNAL"

    # 风控通过后的下单指令：由 RiskManager 在校验通过后发出，由 OrderManager 消费执行
    ORDER   = "ORDER"

    # 成交回报：由 BinanceGateway 从交易所 User Data Stream 接收后发出
    # 携带成交价、数量、订单ID、累计成交量等字段
    FILL    = "FILL"

    # 账户余额更新：Binance ACCOUNT_UPDATE 推送触发，由 AccountManager 同步本地余额缓存
    ACCOUNT = "ACCOUNT"

    # 订单簿快照/增量更新：OrderBookEngine 每次更新后发出，携带 top-N 档买卖盘数据
    # 消费者：高频策略（计算 OFI、盘口压力等微观结构指标）
    ORDER_BOOK = "ORDER_BOOK"

    # ── 决策层事件（终极版新增）──────────────────────────────────────────────

    # 融合 alpha 更新：AlphaFusionEngine 每次 rank 后发出
    # 携带 {symbol: FusedAlpha} 快照，供监控/归因模块使用
    ALPHA_UPDATE = "ALPHA_UPDATE"

    # 目标组合更新：PortfolioConstructor 输出的 TargetPortfolio
    # 携带 {longs, shorts, to_close_long, to_close_short}
    TARGET_POSITION = "TARGET_POSITION"

    # ── 风控事件（终极版新增）────────────────────────────────────────────────

    # 风控拒绝通知：RiskManager 拒绝 SIGNAL 时发出，携带 reject_reason
    RISK_REJECT = "RISK_REJECT"

    # Kill Switch 触发通知：ShockDetector 触发全局暂停时发出
    KILL_SWITCH = "KILL_SWITCH"


# ── Binance 端点常量 ──────────────────────────────────────────────────────────
class BinanceEndpoints:
    """
    集中管理所有 Binance Futures 的 REST 和 WebSocket 端点。

    使用方式：
        from core.constants import BinanceEndpoints as EP
        url = EP.trade_stream("BTCUSDT")
    """

    # REST
    FAPI_BASE        = "https://fapi.binance.com"
    FAPI_DEPTH       = "https://fapi.binance.com/fapi/v1/depth"
    FAPI_EXCHANGE    = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    FAPI_PREMIUM_IDX = "https://fapi.binance.com/fapi/v1/premiumIndex"
    FAPI_OPEN_INT    = "https://fapi.binance.com/fapi/v1/openInterest"
    FAPI_TICKER_24H  = "https://fapi.binance.com/fapi/v1/ticker/24hr"

    # WebSocket 主网
    WS_BASE          = "wss://fstream.binance.com/ws"

    # WebSocket 测试网
    TESTNET_API      = "https://testnet.binancefuture.com"
    TESTNET_WS       = "wss://stream.binancefuture.com"

    @staticmethod
    def agg_trade_stream(symbol: str) -> str:
        return f"{BinanceEndpoints.WS_BASE}/{symbol.lower()}@aggTrade"

    @staticmethod
    def book_ticker_stream(symbol: str) -> str:
        return f"{BinanceEndpoints.WS_BASE}/{symbol.lower()}@bookTicker"

    @staticmethod
    def depth_stream(symbol: str, speed_ms: int = 100) -> str:
        return f"{BinanceEndpoints.WS_BASE}/{symbol.lower()}@depth@{speed_ms}ms"

    @staticmethod
    def user_data_stream(listen_key: str) -> str:
        return f"{BinanceEndpoints.WS_BASE}/{listen_key}"

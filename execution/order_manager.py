"""
订单执行模块（OrderManager）

职责：
    监听经风控验证后的 ORDER 事件，调用交易网关提交市价单到交易所。
    本模块是"执行层"，职责单一：接单并发单，不包含任何风控逻辑。

设计原则：
    风控逻辑完全在 RiskManager 中处理，OrderManager 接收到的 ORDER 事件
    已经过 RiskManager 的全部检查，可以直接提交。
    这种分层设计使风控逻辑和执行逻辑完全解耦，便于独立修改和测试。

在系统事件流中的位置：
    RiskManager（风控通过）
        │  emit ORDER{symbol, side, qty, price}
        ▼
    OrderManager.on_order_event()
        │  调用 gateway.place_market_order()
        ▼
    BinanceGateway（提交到交易所）
        │  收到成交回报后 emit FILL 事件
        ▼
    PositionManager / RiskManager / Database

ORDER 事件 data 字段约定：
    symbol (str)   : 交易对，例如 "BTCUSDT"
    side   (str)   : "BUY" 或 "SELL"
    qty    (float) : 下单数量；若为 0，则由 PositionSizer 自动计算
    price  (float) : 参考价格（qty=0 时用于计算数量，可为 None）
"""

from core.constants import EventType

# qty=0 时的兜底下单量（策略未指定数量时使用）
_DEFAULT_QTY = 0.001


class OrderManager:
    """
    订单执行管理器。

    从 EventEngine 接收 ORDER 事件，通过 gateway 提交市价单，
    支持自动数量计算（当 ORDER 事件中 qty=0 时）。
    """

    def __init__(self, gateway, event_engine):
        """
        初始化订单管理器，注册 ORDER 事件监听。

        参数：
            gateway      (BinanceGateway) : 交易网关，用于实际提交订单
            event_engine (EventEngine)    : 全局事件引擎
        """
        self.gateway = gateway
        self.event_engine = event_engine

        # 注册 ORDER 事件：RiskManager 风控通过后会发出此事件
        event_engine.register(EventType.ORDER, self.on_order_event)

    # -------------------------
    # 事件处理（主路径）
    # -------------------------

    def on_order_event(self, event):
        """
        接收经风控验证的 ORDER 事件，向交易所提交市价单。

        数量决策逻辑：
            1. 若 ORDER 事件中携带了 qty（且 qty > 0），直接使用该数量
               （通常由策略在发出 SIGNAL 时指定，经风控后原样传递）
            2. 若 qty == 0，说明策略只指定了方向，数量由 PositionSizer 计算
               （需要参考价格 price 来计算，例如固定风险金额法需要价格）
            3. 若 qty == 0 且 price 也缺失，则跳过无法处理的订单并打印警告

        参数：
            event (Event) : EventType.ORDER 事件
        """
        data = event.data
        symbol = data["symbol"]
        side = data["side"]

        # 尝试从事件中获取数量，默认 0 表示需要自动计算
        qty = data.get("qty", 0)
        if qty == 0:
            # qty 为 0，需要用 PositionSizer 根据参考价格计算下单量
            price = data.get("price")
            if price is None:
                # 既无 qty 又无 price，无法计算数量，跳过此订单
                print("[OrderManager] ORDER 事件缺少 qty 和 price，跳过")
                return
            # 策略未指定数量，使用默认兜底数量
            qty = _DEFAULT_QTY

        try:
            # 调用网关提交市价单，网关内部会做精度格式化后发送到交易所
            order = self.gateway.place_market_order(symbol, side, qty)
            print(f"[OrderManager] 订单已提交: {symbol} {side} {qty} → {order}")

        except Exception as e:
            # 捕获所有异常（网络错误、API 错误等），打印后继续运行
            # 不重新抛出，避免影响 EventEngine 的事件循环
            print(f"[OrderManager] 下单失败: {e}")

    # -------------------------
    # 手动下单入口（调试/干预用）
    # -------------------------

    def place_order(self, symbol: str, side: str, qty: float):
        """
        直接下单，绕过事件队列，仅供调试或手动干预使用。

        注意：此方法绕过了 RiskManager 的风控检查，
        仅在紧急手动干预时使用（如手动止损、调整持仓）。
        常规交易路径应通过 EventEngine 的 SIGNAL → ORDER 流程。

        参数：
            symbol (str)   : 交易对
            side   (str)   : "BUY" 或 "SELL"
            qty    (float) : 下单数量

        返回：
            dict 或 None : 交易所返回的订单信息，失败时返回 None
        """
        try:
            order = self.gateway.place_market_order(symbol, side, qty)
            print(f"[OrderManager] 手动订单已提交: {symbol} {side} {qty} → {order}")
            return order
        except Exception as e:
            print(f"[OrderManager] 手动下单失败: {e}")

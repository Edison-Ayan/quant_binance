"""
Binance 合约交易网关（BinanceGateway）

职责：
    作为系统与 Binance 合约交易所之间的唯一接口，负责：
    1. 下单（市价单 / 限价单）和撤单
    2. 通过 User Data Stream 监听成交回报（FILL）和账户变动（ACCOUNT）
    3. 将交易所原始推送数据解析并封装为系统事件，投递到 EventEngine

User Data Stream 工作机制：
    - 调用 REST API 获取 listenKey（有效期 60 分钟）
    - 用 listenKey 建立 WebSocket 连接，接收账户级推送（仅本账户可见）
    - 每 30 分钟调用 keepalive 接口延续 listenKey 有效期
    - WebSocket 断线后自动重新获取 listenKey 并重连

精度处理：
    Binance 对每个交易对的下单数量（stepSize）和价格（tickSize）有精度要求，
    使用 Decimal 进行精确的向下取整（ROUND_DOWN），避免浮点误差导致 API 拒单。

在系统事件流中的位置：
    本模块负责两个方向：
    ① 接收方向：交易所推送 → BinanceGateway → EventEngine（FILL / ACCOUNT 事件）
    ② 发送方向：OrderManager 调用 place_market_order → 交易所
"""

import json
import time
import threading
import websocket
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
from binance.exceptions import BinanceAPIException

from core.event import Event
from core.constants import EventType


class BinanceGateway:
    """
    Binance 合约交易网关。

    封装所有与交易所的通信逻辑，对上层业务模块（OrderManager）提供简洁的
    下单接口，同时通过事件机制将成交和账户变动异步通知给系统各模块。
    """

    def __init__(self, api_key, api_secret, event_engine):
        """
        初始化网关，建立 REST 客户端并启动 User Data Stream。

        参数：
            api_key      (str)         : Binance API Key
            api_secret   (str)         : Binance API Secret
            event_engine (EventEngine) : 全局事件引擎，用于投递 FILL / ACCOUNT 事件
        """
        # python-binance REST 客户端，用于下单、查询和维护 listenKey
        self.client = Client(api_key, api_secret)
        self.event_engine = event_engine

        # listenKey：User Data Stream 的鉴权令牌，60 分钟过期，需定期 keepalive
        self.listen_key = None
        # WebSocket 连接对象
        self.ws = None
        # WebSocket 运行线程
        self.ws_thread = None

        # 全局运行标志，False 时停止 keepalive 线程和 WebSocket 重连
        self.running = False

        # 订单回调字典：key=orderId，value=回调函数
        # 用于某些调用方希望在特定订单成交时收到通知的场景
        self.order_callbacks = {}

        # 各交易对的精度信息：key=symbol，value={"step": Decimal, "tick": Decimal}
        # step：数量精度（stepSize），tick：价格精度（tickSize）
        self.symbol_filters = {}

        # 初始化：从交易所拉取所有交易对的精度规则，防止下单时精度不合规被拒绝
        self._load_exchange_info()

        # 启动 User Data Stream（获取 listenKey + 建立 WebSocket + 启动 keepalive）
        self._start_user_stream()

    # --------------------------------
    # 交易所精度信息
    # --------------------------------

    def _load_exchange_info(self):
        """
        拉取并缓存所有合约交易对的数量精度（stepSize）和价格精度（tickSize）。

        为什么需要缓存精度？
            Binance API 对下单数量和价格有严格的精度要求，例如 BTC 数量精度为
            0.001，若传入 0.0015 会被拒绝。通过预缓存精度规则，在下单前
            先做向下取整，确保符合交易所规范。
        """
        # 拉取合约交易所全量信息（包含所有品种的过滤器规则）
        info = self.client.futures_exchange_info()

        for s in info["symbols"]:

            symbol = s["symbol"]

            # 初始化精度为 0（若某品种没有对应过滤器则保持 0，表示无精度限制）
            step = Decimal("0")  # 数量精度（LOT_SIZE 过滤器）
            tick = Decimal("0")  # 价格精度（PRICE_FILTER 过滤器）

            for f in s["filters"]:

                if f["filterType"] == "LOT_SIZE":
                    # LOT_SIZE 过滤器定义下单数量的最小变动单位（stepSize）
                    # 例如 stepSize="0.001" 表示数量必须是 0.001 的整数倍
                    step = Decimal(f["stepSize"])

                if f["filterType"] == "PRICE_FILTER":
                    # PRICE_FILTER 过滤器定义价格的最小变动单位（tickSize）
                    # 例如 tickSize="0.01" 表示价格必须是 0.01 的整数倍
                    tick = Decimal(f["tickSize"])

            # 以 symbol 为键缓存精度规则，供下单时使用
            self.symbol_filters[symbol] = {
                "step": step,
                "tick": tick
            }

    # --------------------------------
    # 精度格式化
    # --------------------------------

    def _format_qty(self, symbol, qty):
        """
        将下单数量向下取整到符合交易所要求的精度。

        为什么用整除而不是 round()？
            round() 会做四舍五入，可能导致结果超过原始值（如 0.0015 → 0.002），
            从而超出账户可用余额。向下取整（floor）始终安全，不会超额下单。

        使用 Decimal 而不是 float？
            float 存在精度问题（如 0.1 + 0.2 ≠ 0.3），Decimal 可以精确表示
            十进制小数，避免精度累积误差导致格式化结果出错。

        参数：
            symbol (str)   : 交易对
            qty    (float) : 原始数量

        返回：
            float : 取整后的合规数量
        """
        step = self.symbol_filters[symbol]["step"]

        # (qty // step) * step：先做整除（去掉不满足精度的尾数），再乘回来
        return float((Decimal(qty) // step) * step)

    def _format_price(self, symbol, price):
        """
        将价格向下取整到符合交易所要求的精度。

        同 _format_qty，使用 Decimal 整除保证精度正确性，向下取整防止超价。

        参数：
            symbol (str)   : 交易对
            price  (float) : 原始价格

        返回：
            float : 取整后的合规价格
        """
        tick = self.symbol_filters[symbol]["tick"]

        return float((Decimal(price) // tick) * tick)

    # --------------------------------
    # User Data Stream 管理
    # --------------------------------

    def _start_user_stream(self):
        """
        启动 User Data Stream：获取 listenKey，建立 WebSocket，启动 keepalive 线程。

        User Data Stream 是 Binance 账户级别的私有推送通道，只传递本账户的
        订单更新（ORDER_TRADE_UPDATE）和账户变动（ACCOUNT_UPDATE），
        不同于行情 WebSocket（公开数据）。
        """
        # 通过 REST 接口生成 listenKey（有效期 60 分钟）
        self.listen_key = self.client.futures_stream_get_listen_key()

        self.running = True

        # 用 listenKey 建立私有 WebSocket 连接
        self._connect_ws()

        # 启动 keepalive 守护线程：每 30 分钟延续 listenKey 有效期
        # 若不定期续期，listenKey 60 分钟后失效，WebSocket 会被交易所断开
        threading.Thread(
            target=self._keepalive_listen_key,
            daemon=True
        ).start()

    # --------------------------------

    def _keepalive_listen_key(self):
        """
        定期续期 listenKey，防止 User Data Stream 因超时而断开。

        Binance 规定 listenKey 有效期为 60 分钟，每次调用 keepalive 接口
        可重置计时器到 60 分钟。每 30 分钟调用一次（安全冗余）。

        在独立守护线程中运行，不阻塞主流程。
        """
        while self.running:

            try:

                # 调用 REST 接口续期 listenKey（PUT /fapi/v1/listenKey）
                self.client.futures_stream_keepalive(self.listen_key)

            except Exception as e:

                print("listen_key keepalive失败:", e)

            # 每 30 分钟（1800 秒）续期一次，远低于 60 分钟过期时间
            time.sleep(30 * 60)

    # --------------------------------
    # WebSocket 连接管理
    # --------------------------------

    def _connect_ws(self):
        """
        建立 User Data Stream 的 WebSocket 连接。

        连接地址使用 listenKey 作为路径参数，交易所通过此 key 识别账户，
        推送专属的订单和账户更新消息。

        ping_interval=20 和 ping_timeout=10：定期发送 ping 保持连接活跃，
        防止网络设备（路由器/NAT）因空闲超时而关闭连接。
        """
        # User Data Stream 连接地址，listenKey 嵌入 URL 路径（私有鉴权方式）
        url = f"wss://fstream.binance.com/ws/{self.listen_key}"

        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,       # 连接成功回调
            on_message=self._on_message,  # 消息接收回调
            on_error=self._on_error,      # 错误处理回调
            on_close=self._on_close       # 连接关闭回调（含自动重连逻辑）
        )

        # 在独立守护线程中运行 WebSocket 循环
        self.ws_thread = threading.Thread(
            target=lambda: self.ws.run_forever(
                ping_interval=20,   # 每 20 秒发送一次 ping
                ping_timeout=10     # 10 秒内未收到 pong 则视为超时
            )
        )

        self.ws_thread.daemon = True
        self.ws_thread.start()

    # --------------------------------

    def _on_open(self, ws):
        """
        WebSocket 连接成功回调。记录连接建立，无需额外操作（User Data Stream
        连接建立即自动开始推送，无需发送订阅消息）。
        """
        print("WebSocket connected")

    # --------------------------------

    def _on_message(self, ws, message):
        """
        接收并路由 User Data Stream 的推送消息。

        Binance User Data Stream 消息类型（"e" 字段）：
            ORDER_TRADE_UPDATE : 订单状态变化（新单、部分成交、完全成交、撤单等）
            ACCOUNT_UPDATE     : 账户余额和持仓变动

        参数：
            ws      : WebSocketApp 实例
            message : 原始 JSON 字符串
        """
        data = json.loads(message)

        # "e" 字段标识消息类型，据此分发到对应处理函数
        event_type = data.get("e")

        if event_type == "ORDER_TRADE_UPDATE":
            # 订单状态更新：解析成交信息并发布 FILL 事件
            self._handle_order_update(data)

        elif event_type == "ACCOUNT_UPDATE":
            # 账户余额/持仓更新：解析并发布 ACCOUNT 事件
            self._handle_account_update(data)

    # --------------------------------
    # 订单成交处理
    # --------------------------------

    def _handle_order_update(self, data):
        """
        解析 ORDER_TRADE_UPDATE 消息，封装为 FILL 事件并投递到 EventEngine。

        Binance ORDER_TRADE_UPDATE 消息中 "o" 子对象字段说明：
            s  (symbol)         : 交易对，例如 "BTCUSDT"
            S  (side)           : 方向，"BUY" 或 "SELL"
            X  (currentStatus)  : 订单当前状态，常见值：
                                      NEW（新建）、PARTIALLY_FILLED（部分成交）、
                                      FILLED（完全成交）、CANCELED（已撤销）
            i  (orderId)        : 交易所分配的订单 ID（整数），全局唯一
            l  (lastFilledQty)  : 本次成交数量（本条消息触发的那笔成交量）
                                  注意：一个订单可能分多次成交，每次触发一条消息
            z  (cumFilledQty)   : 订单累计已成交数量（从下单到目前为止的总成交量）
                                  当 z == 原始下单数量时，说明订单完全成交
            ap (avgPrice)       : 订单当前平均成交价格（加权均价，随每笔成交更新）
                                  注意：不是本次成交价，而是迄今所有成交的均价

        参数：
            data (dict) : 已解析的 ORDER_TRADE_UPDATE 消息
        """
        # "o" 子对象包含订单的所有详细字段
        order = data["o"]

        # s：交易对（symbol），例如 "BTCUSDT"
        symbol = order["s"]

        # S：交易方向（side），"BUY" 或 "SELL"
        side = order["S"]

        # X：订单当前状态（currentOrderStatus）
        # FILLED 表示完全成交，PositionManager 等模块可据此判断是否更新持仓
        status = order["X"]

        # i：交易所分配的订单 ID（orderId），用于关联本地订单记录和回调函数
        order_id = order["i"]

        # l：本次成交数量（lastFilledQuantity），即触发本条推送的那笔成交的数量
        # 一个大订单可能分多次成交，每次成交都推送一条消息，l 是本次的增量
        last_qty = float(order["l"])

        # z：订单累计成交数量（cumulativeFilledQuantity），从下单至今的总成交量
        # 可用于判断订单完成进度：z / 原始下单量 = 成交比例
        cum_qty = float(order["z"])

        # ap：订单平均成交价格（averagePrice）
        # 这是加权均价而非最后一笔成交价，对于分批成交的订单更能反映实际成本
        price = float(order["ap"])

        # 封装 FILL 事件，包含持仓管理、风控和数据库所需的所有字段
        event = Event(
            EventType.FILL,
            {
                "symbol": symbol,
                "side": side,
                "order_id": order_id,
                "status": status,
                "last_qty": last_qty,  # 本次成交量（增量）
                "cum_qty": cum_qty,    # 累计成交量
                "price": price         # 订单均价（ap 字段）
            }
        )

        # 将 FILL 事件投递到事件引擎，通知 PositionManager / RiskManager / Database
        self.event_engine.put(event)

        # 若该订单注册了回调函数，则同步调用（用于需要感知特定订单状态的场景）
        cb = self.order_callbacks.get(order_id)

        if cb:
            cb(event)

    # --------------------------------
    # 账户余额处理
    # --------------------------------

    def _handle_account_update(self, data):
        """
        解析 ACCOUNT_UPDATE 消息，封装为 ACCOUNT 事件并投递到 EventEngine。

        Binance ACCOUNT_UPDATE 消息结构：
            data["a"]["B"] : 余额变化列表，每项格式：
                             {"a": "USDT", "wb": "1000.00", "cw": "990.00"}
                             a  = asset（资产名称）
                             wb = walletBalance（钱包余额）
                             cw = crossWalletBalance（全仓余额）
            data["a"]["P"] : 持仓变化列表，每项包含持仓数量、入场价格等

        参数：
            data (dict) : 已解析的 ACCOUNT_UPDATE 消息
        """
        # "a" 子对象包含余额和持仓的变化详情
        account = data["a"]

        # "B"：余额变化列表（Balances）
        balances = account["B"]

        # "P"：持仓变化列表（Positions）
        positions = account["P"]

        # 封装 ACCOUNT 事件，由 AccountManager 消费同步本地余额缓存
        event = Event(
            EventType.ACCOUNT,
            {
                "balances": balances,
                "positions": positions
            }
        )

        self.event_engine.put(event)

    # --------------------------------
    # WebSocket 错误处理
    # --------------------------------

    def _on_error(self, ws, error):
        """
        WebSocket 错误回调。打印错误信息，等待 _on_close 触发重连。
        """
        print("WebSocket error:", error)

    # --------------------------------

    def _on_close(self, ws, code, msg):
        """
        WebSocket 关闭回调，实现自动重连机制。

        为什么需要重连？
            网络波动、服务器维护或 listenKey 过期都可能导致 WebSocket 断开。
            成交回报的丢失会导致持仓状态错误，因此必须尽快重连。
            重连时重新获取新的 listenKey（旧的可能已失效），然后重建连接。

        参数：
            code : WebSocket 关闭状态码
            msg  : 关闭原因描述
        """
        print("WebSocket closed")

        # 只有在系统仍在运行时才重连（调用 close() 主动关闭时不重连）
        if self.running:

            # 等待 5 秒再重连，避免因瞬时网络抖动导致频繁重连冲击服务器
            time.sleep(5)

            # 重新获取 listenKey（旧 key 可能已失效）
            self.listen_key = self.client.futures_stream_get_listen_key()

            # 重建 WebSocket 连接
            self._connect_ws()

    # --------------------------------
    # 下单接口
    # --------------------------------

    def place_market_order(
        self,
        symbol,
        side,
        qty,
        callback=None
    ):
        """
        提交合约市价单。

        市价单会以当前市场最优价格立即成交，适合需要快速执行的场景。
        提交前先对数量做精度格式化，确保符合交易所规范。

        参数：
            symbol   (str)      : 交易对，例如 "BTCUSDT"
            side     (str)      : "BUY" 或 "SELL"
            qty      (float)    : 下单数量（原始值，内部会自动格式化）
            callback (Callable) : 可选的成交回调函数，在收到 FILL 事件时调用

        返回：
            dict : 交易所返回的订单信息（包含 orderId 等），失败时返回 None
        """
        # 将数量向下取整到合规精度，避免 API 因精度不符而拒单
        qty = self._format_qty(symbol, qty)

        try:

            # 调用 Binance 合约下单接口（/fapi/v1/order），type="MARKET" 为市价单
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=qty
            )

            # 提取订单 ID，用于关联后续的成交回报和回调函数
            order_id = order["orderId"]

            # 若调用方传入了回调函数，注册到回调字典，等待成交推送时触发
            if callback:

                self.order_callbacks[order_id] = callback

            return order

        except BinanceAPIException as e:

            # API 异常（如余额不足、精度错误、超过持仓限制等）打印错误后返回 None
            print("order error:", e)

    # --------------------------------

    def place_limit_order(
        self,
        symbol,
        side,
        qty,
        price,
        tif="GTC",
        callback=None
    ):
        """
        提交合约限价单。

        限价单以指定价格挂单，不立即成交，等待市场价格到达后成交。
        适合对成本敏感、愿意等待成交的场景。

        参数：
            symbol   (str)      : 交易对
            side     (str)      : "BUY" 或 "SELL"
            qty      (float)    : 下单数量
            price    (float)    : 限价价格
            tif      (str)      : 有效期类型（Time In Force）：
                                     GTC = Good Till Cancel（撤销前有效，默认）
                                     IOC = Immediate Or Cancel（立即成交或撤销）
                                     FOK = Fill Or Kill（全部成交或全部撤销）
            callback (Callable) : 可选的成交回调函数

        返回：
            dict : 交易所返回的订单信息，失败时返回 None
        """
        # 数量和价格都需要格式化，确保符合精度规范
        qty = self._format_qty(symbol, qty)

        price = self._format_price(symbol, price)

        try:

            # 调用限价单接口，price 需传字符串格式（避免 JSON 浮点精度丢失）
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                quantity=qty,
                price=str(price),  # 字符串格式，防止浮点精度问题
                timeInForce=tif
            )

            order_id = order["orderId"]

            if callback:

                self.order_callbacks[order_id] = callback

            return order

        except BinanceAPIException as e:

            print("limit order error:", e)

    # --------------------------------
    # 撤单接口
    # --------------------------------

    def cancel_order(self, symbol, order_id):
        """
        撤销指定订单。

        适用于限价单未成交时的主动取消，或止损策略中的条件撤单。

        参数：
            symbol   (str) : 交易对
            order_id (int) : 要撤销的订单 ID（由 place_limit_order 返回）

        返回：
            dict : 撤单结果，失败时返回 None
        """
        try:

            return self.client.futures_cancel_order(
                symbol=symbol,
                orderId=order_id
            )

        except BinanceAPIException as e:

            print("cancel error:", e)

    # --------------------------------
    # 查询订单
    # --------------------------------

    def get_order(self, symbol, order_id):
        """
        查询指定订单的当前状态。

        可用于主动轮询订单成交进度（补偿 WebSocket 推送可能的漏报）。

        参数：
            symbol   (str) : 交易对
            order_id (int) : 订单 ID

        返回：
            dict : 订单详情（含状态、成交数量、均价等），失败时返回 None
        """
        try:

            return self.client.futures_get_order(
                symbol=symbol,
                orderId=order_id
            )

        except BinanceAPIException as e:

            print("get order error:", e)

    # --------------------------------
    # 关闭网关
    # --------------------------------

    def close(self):
        """
        优雅关闭网关：停止重连逻辑并关闭 WebSocket 连接。

        设置 running=False 后，_on_close 回调中的重连逻辑不会触发，
        _keepalive_listen_key 线程会在下次循环后退出。
        """
        # 标记停止，防止 _on_close 触发自动重连
        self.running = False

        if self.ws:

            # 主动关闭 WebSocket 连接
            self.ws.close()

"""
持仓管理模块（PositionManager）

职责：
    维护系统内所有交易品种的实时持仓状态，包括：
    1. 接收 FILL 事件，根据成交方向和数量更新持仓（加仓/减仓/平仓）
    2. 接收 TICK 事件，实时更新浮动盈亏（unrealized_pnl）
    3. 对外提供持仓查询接口，供风控、监控等模块使用

Position.update() 加权均价公式说明：
    加仓时，新的持仓成本（入场均价）需要反映历史仓位和新仓位的综合成本，
    使用加权平均价：
        新均价 = (旧均价 × |旧数量| + 新成交价 × |新成交数量|) / |新总数量|
    例如：已持有 1 BTC @ 40000，再买 1 BTC @ 42000：
        新均价 = (40000×1 + 42000×1) / 2 = 41000

    减仓时（同向反转，即卖出已有多头仓位）：
        只减少数量，不更新入场均价。
        原因：已实现的盈亏不再影响剩余仓位的成本基础。
        例如：持有 2 BTC @ 41000，卖出 1 BTC @ 43000：
            剩余 1 BTC 的入场均价仍为 41000，已实现盈亏 = (43000-41000)×1 = 2000

在系统事件流中的位置：
    FILL  → on_fill()  → 更新持仓数量和入场均价
    TICK  → on_tick()  → 更新持仓浮盈亏
"""

from core.constants import EventType


class Position:
    """
    单个交易品种的持仓状态对象。

    维护该品种的：
    - qty：当前净持仓数量（正数=多头，负数=空头，0=无仓位）
    - entry_price：持仓的加权平均入场价格（成本价）
    - unrealized_pnl：当前浮动盈亏（基于最新 TICK 价格实时计算）
    """

    def __init__(self, symbol):
        """
        初始化持仓对象，所有初始值为零（无仓位状态）。

        参数：
            symbol (str) : 交易对，例如 "BTCUSDT"
        """
        self.symbol = symbol

        # 净持仓数量：正值=多头，负值=空头，0=无仓位
        self.qty = 0.0

        # 加权平均入场价格（持仓成本价），无仓位时为 0
        self.entry_price = 0.0

        # 未实现盈亏：(当前价格 - 入场均价) × 持仓数量
        # 每次收到 TICK 事件时由 PositionManager.on_tick() 更新
        self.unrealized_pnl = 0.0

        # ── Alpha 生命周期追踪字段 ─────────────────────────────────────────────
        # 当前 Alpha 生命周期状态（BUILD / EXPANSION / DECAY / REVERSAL）
        self.alpha_state: str = ""
        # 建仓时的 unified alpha 分数（用于事后归因）
        self.entry_alpha: float = 0.0
        # 持仓期间观测到的峰值 unified alpha
        self.peak_alpha: float = 0.0
        # 持仓原因标签（strategy_open / lifecycle_reversal 等）
        self.holding_reason: str = ""
        # 最近一次生命周期状态更新的时间戳（time.time()）
        self.last_lifecycle_update: float = 0.0

    def update(self, qty, price):
        """
        根据成交信息更新持仓状态。

        参数：
            qty   (float) : 成交数量，正数=买入，负数=卖出
            price (float) : 本次成交均价

        逻辑分支：
            1. 加仓（同向操作）：qty 与 self.qty 同号，或当前无仓位
               使用加权平均价更新 entry_price，数量叠加
            2. 减仓（反向操作）：qty 与 self.qty 异号，绝对值 < 当前持仓
               只减少数量，不修改 entry_price（已实现部分不影响剩余成本）
            3. 平仓（减至零）：数量变为 0，同时清零 entry_price
        """
        # 判断是加仓还是减仓：
        # self.qty * qty >= 0 时，两者同号（或其中一个为零），表示加仓或初始建仓
        # 例如：self.qty=1，qty=0.5（都是正数），是加仓
        # 例如：self.qty=0，qty=1（初次建仓），也走加仓逻辑
        if self.qty * qty >= 0:

            # 加仓路径：计算加仓后的新总数量
            new_qty = self.qty + qty

            if new_qty != 0:
                # ─────────────────────────────────────────────────────────
                # 加权平均入场价公式：
                #   新均价 = (旧均价 × |旧数量| + 成交价 × |新成交数量|) / |新总数量|
                #
                # 使用绝对值（abs）的原因：
                #   空头仓位的 qty 为负数，若不取绝对值，负数乘以负数会变正，
                #   导致公式计算错误。取绝对值后统一按"份额"加权，方向信息
                #   由 qty 的符号体现，均价始终为正数。
                #
                # 示例（多头加仓）：
                #   已持 1 BTC @ 40000，再买 1 BTC @ 42000：
                #   新均价 = (40000×1 + 42000×1) / 2 = 41000.0
                #
                # 示例（空头加仓）：
                #   已空 1 BTC @ 45000，再空 1 BTC @ 43000（qty=-1）：
                #   新均价 = (45000×|-1| + 43000×|-1|) / |-2| = 44000.0
                # ─────────────────────────────────────────────────────────
                self.entry_price = (
                    self.entry_price * abs(self.qty)  # 旧仓位的总成本（价格×数量）
                    + price * abs(qty)                # 新成交的成本
                ) / abs(new_qty)                      # 除以新总数量得加权均价

            # 更新持仓数量为加仓后的新总量
            self.qty = new_qty

        else:

            # 减仓路径：qty 与 self.qty 符号相反，表示平减持仓
            # 只修改数量，不更新入场均价
            # 原因：剩余仓位的成本基础未变，已平仓部分的盈亏由
            # RiskManager.on_fill_event() 单独计算
            self.qty += qty

            # 若减仓后持仓归零（完全平仓），清除入场价，回到空仓状态
            if self.qty == 0:

                self.entry_price = 0


class PositionManager:
    """
    全品种持仓管理器。

    维护所有交易品种的 Position 对象字典，并通过事件注册机制自动监听
    成交（FILL）和行情（TICK）事件，实时保持持仓状态准确。
    """

    def __init__(self, event_engine):
        """
        初始化持仓管理器，并向事件引擎注册 FILL 和 TICK 事件监听。

        参数：
            event_engine (EventEngine) : 全局事件引擎
        """
        self.event_engine = event_engine

        # 持仓字典：key=symbol，value=Position 对象
        # 按需创建，首次收到某品种成交时才初始化该品种的 Position
        self.positions = {}

        # 最新价格缓存：key=symbol，value=最新成交价
        # 用于计算浮盈亏，以及供外部模块查询当前市价
        self.last_price = {}

        # 注册 FILL 事件：每次成交后更新持仓数量和均价
        event_engine.register(
            EventType.FILL,
            self.on_fill
        )

        # 注册 TICK 事件：每次行情更新后重新计算浮盈亏
        event_engine.register(
            EventType.TICK,
            self.on_tick
        )

    # --------------------------
    # 成交事件处理
    # --------------------------

    def on_fill(self, event):
        """
        处理成交事件，更新对应品种的持仓状态。

        从 FILL 事件中提取成交方向、数量和价格，
        调用 Position.update() 更新持仓。BUY 传正数量，SELL 传负数量。

        参数：
            event (Event) : EventType.FILL 事件，data 字段包含成交详情
        """
        data = event.data

        symbol = data["symbol"]
        side = data["side"]

        # 本次成交的均价（Binance ap 字段，即订单平均成交价）
        price = data["price"]
        # 本次成交数量（Binance l 字段，即本条消息触发的成交增量）
        qty = data["last_qty"]

        # 若该品种尚无持仓记录，则初始化一个空的 Position 对象
        if symbol not in self.positions:

            self.positions[symbol] = Position(symbol)

        if side == "BUY":
            # 买入：传正数量，表示多头方向
            self.positions[symbol].update(qty, price)

        else:
            # 卖出：传负数量，表示空头方向或平多仓
            self.positions[symbol].update(-qty, price)

    # --------------------------
    # 行情事件处理
    # --------------------------

    def on_tick(self, event):
        """
        处理逐笔行情事件，更新持仓的浮动盈亏。

        浮盈亏计算公式：
            unrealized_pnl = (当前价格 - 入场均价) × 持仓数量

        多头（qty > 0）：价格上涨则 pnl > 0，价格下跌则 pnl < 0
        空头（qty < 0）：价格下跌则 pnl > 0（两个负数相乘为正），价格上涨则 pnl < 0

        参数：
            event (Event) : EventType.TICK 事件，data 包含 symbol 和最新价格
        """
        symbol = event.data["symbol"]

        price = event.data["price"]

        # 缓存最新价格，供其他模块（如风控）查询市价使用
        self.last_price[symbol] = price

        # 获取该品种的持仓对象（若无持仓则返回 None）
        pos = self.positions.get(symbol)

        # 只有存在持仓（qty != 0）时才计算浮盈亏，避免除零或无意义计算
        if pos and pos.qty != 0:

            # (当前价格 - 入场均价) × 净持仓数量
            # 多头：qty > 0，价格涨则盈，跌则亏
            # 空头：qty < 0，价格跌则 (price - entry) < 0，乘以负 qty 得正值（盈利）
            pos.unrealized_pnl = (
                price - pos.entry_price
            ) * pos.qty

    # --------------------------
    # 查询接口
    # --------------------------

    def get_position(self, symbol):
        """
        获取指定品种的持仓对象。

        返回：
            Position 或 None（若该品种从未成交过）
        """
        return self.positions.get(symbol)

    def get_total_exposure(self):
        """
        计算所有品种的总名义敞口（各品种持仓数量绝对值之和）。

        这里的敞口是数量维度（非 USDT 价值），若需要价值维度需乘以各品种价格。
        风控模块可用此值判断整体仓位是否过重。

        返回：
            float : 所有品种持仓数量绝对值的总和
        """
        exposure = 0

        for p in self.positions.values():

            # abs(p.qty)：无论多空，都计入敞口（方向相消会低估风险）
            exposure += abs(p.qty)

        return exposure

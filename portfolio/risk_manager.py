"""
风险管理模块（RiskManager）

职责：
    在策略信号（SIGNAL）转化为实际下单（ORDER）之前，执行多道风控检查，
    过滤掉可能造成过度损失或违规操作的信号，保护账户资金安全。

在系统事件流中的位置：
    SIGNAL → on_signal_event() → 三重风控检查 → 通过则 emit ORDER（失败则丢弃）
    FILL   → on_fill_event()   → 更新当日已实现 PnL（用于日亏损检查）

三道风控门（按顺序检查，任意一道失败则拒绝）：
    1. _check_position_limit  : 仓位上限检查（防止单品种过度集中）
    2. _check_order_frequency : 下单频率检查（防止策略 bug 导致刷单）
    3. _check_daily_loss      : 日亏损熔断（触发后关闭当日所有交易）
"""

import time

from core.event import Event
from core.constants import EventType
from data_layer.logger import logger


class RiskManager:
    """
    量化交易风险管理器。

    串联三道风控门：仓位限制、频率限制、日亏损熔断。
    所有策略信号必须经过此模块校验才能转化为实际订单。
    """

    def __init__(self, position_manager, event_engine):
        """
        初始化风控模块，设置风控参数并注册事件监听。

        参数：
            position_manager (PositionManager) : 用于查询当前持仓状态
            event_engine     (EventEngine)     : 用于订阅 SIGNAL/FILL 事件和发出 ORDER 事件
        """
        self.position_manager = position_manager
        self.event_engine = event_engine

        # ── 风控参数 ──────────────────────────────────────────────────
        # 单品种最大仓位（数量单位，非 USDT），防止单个品种仓位过重
        # 注意：动态 qty 模式下每笔约 0.71 BTC（10000U×5%×100x/70000），
        # 最多叠仓 10 笔（max_notional_ratio=50%/risk=5%），合计约 7 BTC，
        # 此处设 1000 作为宽松上限，实际仓位上限由策略的 max_notional_ratio 控制
        self.max_position = 1000

        # 最大日亏损（USDT），达到此阈值后停止当日所有交易
        # 设为负数，因为亏损表现为负的 PnL
        self.max_daily_loss = -5000

        # 每分钟最大下单次数，防止策略 bug 导致频繁下单消耗手续费或触发交易所限频
        # HFT 策略每秒可能产生多个信号，设为 600（10次/秒）；
        # 若需更严格保护，可通过外部配置覆盖此值
        self.max_orders_per_min = 600
        # ─────────────────────────────────────────────────────────────

        # 当日已实现盈亏（USDT），由 on_fill_event() 在每次成交后更新
        self.daily_pnl = 0.0

        # 下单时间戳列表：记录最近下单的时间点（UNIX 时间戳，秒）
        # 用于计算滑动 1 分钟窗口内的下单次数
        self.order_timestamps = []

        # 交易使能标志：False 时拒绝所有交易（日亏损熔断触发后设为 False）
        self.trading_enabled = True

        # 注册事件监听
        # SIGNAL 事件：策略产生信号后，风控模块负责审核
        event_engine.register(EventType.SIGNAL, self.on_signal_event)
        # FILL 事件：成交后更新日盈亏，为日亏损检查提供数据
        event_engine.register(EventType.FILL, self.on_fill_event)

    # -------------------------
    # 事件处理
    # -------------------------

    def on_signal_event(self, event):
        """
        接收策略信号，通过风控后 emit ORDER 事件。

        这是系统中信号→订单转化的关键节点：
        - 策略只负责判断"该不该交易"（信号逻辑）
        - 风控负责判断"能不能交易"（资金和风险约束）
        - 两者分离，避免策略代码混入风控逻辑，提高可维护性

        ORDER 事件将由 OrderManager 消费并提交到交易所。

        参数：
            event (Event) : EventType.SIGNAL 事件
        """
        data = event.data
        symbol = data["symbol"]
        side = data["side"]
        qty = data.get("qty", 0)

        # 经过三重风控检查，全部通过才发出 ORDER 事件
        if self.check_order(symbol, side, qty):
            order_event = Event(
                EventType.ORDER,
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": data.get("price"),
                },
            )
            # 将验证通过的订单指令投递到事件队列，由 OrderManager 执行
            self.event_engine.put(order_event)

    def on_fill_event(self, event):
        """
        接收成交事件，更新当日已实现 PnL。

        为什么只在减仓（平仓）时计算盈亏？
            加仓时只是建立或增加持仓，没有"实现"任何盈亏；
            减仓时，持仓被平掉，盈亏才真正发生并影响账户净值。

        计算逻辑：
            - SELL 成交（平多头）：盈亏 = (成交价 - 入场均价) × 成交数量
            - BUY 成交且当前持空头（平空头）：盈亏 = (入场均价 - 成交价) × 成交数量

        参数：
            event (Event) : EventType.FILL 事件
        """
        data = event.data

        symbol = data["symbol"]
        side = data["side"]
        fill_price = data.get("price", 0.0)    # 本次成交均价
        fill_qty = data.get("last_qty", 0.0)   # 本次成交数量

        # 查询该品种在成交前的持仓状态（用于获取入场均价）
        pos = self.position_manager.get_position(symbol)

        # 只有存在持仓、有实际成交量、且入场均价有效时才计算盈亏
        if pos and fill_qty > 0 and pos.entry_price > 0:
            if side == "SELL":
                # 卖出平多头：(成交价 - 入场均价) × 数量 = 本次实现盈亏
                self.update_pnl((fill_price - pos.entry_price) * fill_qty)
            elif side == "BUY" and pos.qty < 0:
                # 买入平空头（pos.qty < 0 说明当前持有空头仓位）：
                # 空头盈利方向与多头相反，故用 (入场均价 - 成交价)
                self.update_pnl((pos.entry_price - fill_price) * fill_qty)

    # -------------------------
    # 核心检查入口
    # -------------------------

    def check_order(self, symbol: str, side: str, qty: float) -> bool:
        """
        依次执行所有风控检查，全部通过才返回 True。

        短路逻辑：任意一道检查失败则立即返回 False，后续检查不再执行，
        避免不必要的计算（特别是 _check_order_frequency 会修改状态）。

        参数：
            symbol (str)   : 交易对
            side   (str)   : "BUY" 或 "SELL"
            qty    (float) : 下单数量

        返回：
            bool : True=通过所有风控，False=被任意一道风控拒绝
        """
        # 全局交易开关：日亏损熔断触发后，所有订单都被拒绝
        if not self.trading_enabled:
            logger.debug("[RiskManager] 交易已关闭")
            return False

        # 第一道：仓位上限检查
        if not self._check_position_limit(symbol, side, qty):
            return False

        # 第二道：下单频率检查
        if not self._check_order_frequency():
            return False

        # 第三道：日亏损检查
        if not self._check_daily_loss():
            return False

        return True

    # -------------------------
    # 风控门1：仓位限制
    # -------------------------

    def _check_position_limit(self, symbol: str, side: str, qty: float) -> bool:
        """
        检查本次下单后的仓位是否超过单品种最大仓位上限。

        风控原理：
            单品种仓位集中度过高会放大特定品种价格波动对账户的冲击。
            例如 BTC 仓位占比 90%，一旦 BTC 大跌 20%，账户净值损失 18%。
            通过限制单品种最大仓位（绝对数量），控制个股/个币风险敞口。

        实现逻辑：
            计算下单后的预期净仓位（future），若绝对值超过阈值则拒绝。
            正值=多头，负值=空头，均须受约束（防止极端空头仓位同样危险）。

        参数：
            symbol (str)   : 交易对
            side   (str)   : "BUY" 或 "SELL"
            qty    (float) : 本次下单数量

        返回：
            bool : True=仓位未超限，False=超限被拒绝
        """
        pos = self.position_manager.get_position(symbol)
        # 当前净仓位，若从未成交过则视为 0
        current = pos.qty if pos else 0.0

        # 预测下单后的净仓位
        # BUY 增加净仓位（多头方向），SELL 减少净仓位（空头方向）
        future = current + qty if side == "BUY" else current - qty

        if abs(future) > self.max_position:
            logger.debug(f"[RiskManager] {symbol} 超过最大仓位限制 "
                         f"({abs(future):.4f} > {self.max_position})")
            return False

        return True

    # -------------------------
    # 风控门2：下单频率
    # -------------------------

    def _check_order_frequency(self) -> bool:
        """
        检查过去 1 分钟内的下单次数是否超过频率上限。

        风控原理：
            正常策略在 1 分钟内下单次数有限，异常高频下单通常意味着：
            1. 策略 bug（如信号计算出错，每个 Tick 都发信号）
            2. 市场异常（价格剧烈震荡导致均线频繁交叉）
            频繁下单会快速消耗手续费，且可能触发交易所的 API 频率限制（429 Too Many Requests）。

        实现：滑动时间窗口（Sliding Window）
            维护一个时间戳列表，每次检查前先清除 1 分钟前的旧记录，
            然后统计剩余记录数量（即过去 1 分钟内的下单次数）。
            时间复杂度 O(n)，n 为窗口内下单次数（通常很小，可接受）。

        返回：
            bool : True=频率未超限，False=频率超限被拒绝
        """
        now = time.time()

        # 过滤掉 1 分钟（60秒）前的旧时间戳，保留窗口内的记录
        # 这里重新赋值而不是 in-place 修改，保证线程安全（替换引用是原子操作）
        self.order_timestamps = [
            t for t in self.order_timestamps if now - t < 60
        ]

        # 检查窗口内已有的下单次数是否达到上限
        if len(self.order_timestamps) >= self.max_orders_per_min:
            logger.debug(f"[RiskManager] 订单频率过高 ({len(self.order_timestamps)}/min)")
            return False

        # 通过检查后，将当前时间戳记录到窗口中，用于后续计数
        self.order_timestamps.append(now)
        return True

    # -------------------------
    # 风控门3：日亏损熔断
    # -------------------------

    def _check_daily_loss(self) -> bool:
        """
        检查当日累计亏损是否触发熔断阈值。

        风控原理：
            日亏损限制是量化交易最重要的风控手段之一，也称"日内止损线"或"熔断器"。
            无论策略逻辑多么完善，极端市场行情（黑天鹅）、策略 bug 或网络问题
            都可能导致连续亏损。设置日亏损上限可以在损失扩大前强制停机：
            1. 触发熔断后停止当日交易，减少损失
            2. 人工介入检查，排除系统故障或策略失效
            3. 次日手动重置后恢复交易（通过 reset() 方法）

        实现：
            当 daily_pnl <= max_daily_loss（负数阈值）时触发熔断，
            同时将 trading_enabled 设为 False，后续所有信号在第一道检查即被拒绝。

        返回：
            bool : True=日亏损未超限，False=超限触发熔断，停止交易
        """
        if self.daily_pnl <= self.max_daily_loss:
            logger.warning(f"[RiskManager] 超过最大日亏损 {self.daily_pnl:.2f}，停止交易")
            # 关闭交易开关，后续所有订单在 check_order() 第一步就被拒绝
            self.trading_enabled = False
            return False

        return True

    # -------------------------
    # PnL 更新
    # -------------------------

    def update_pnl(self, pnl: float):
        """
        累加当日已实现盈亏。由 on_fill_event() 在每次平仓时调用。

        参数：
            pnl (float) : 本次成交带来的已实现盈亏（正=盈利，负=亏损）
        """
        self.daily_pnl += pnl

    # -------------------------
    # 手动重置（次日使用）
    # -------------------------

    def reset(self):
        """
        重置风控状态，通常在每日开盘前或人工干预后调用。

        重置内容：
            - daily_pnl 归零（新的一天重新计算日盈亏）
            - trading_enabled 恢复为 True（解除熔断）
            - order_timestamps 清空（下单频率窗口清零）

        注意：此方法需人工调用，系统不会自动在午夜重置，
        以防止自动重置掩盖连续亏损的问题。
        """
        self.daily_pnl = 0.0
        self.trading_enabled = True
        self.order_timestamps.clear()
        logger.info("[RiskManager] 风控状态已重置")

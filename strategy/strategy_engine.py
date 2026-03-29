"""
StrategyEngine - 策略生命周期管理

职责：
    管理多个策略实例的注册、启动、停止。
    将 EventEngine 的事件分发给已注册的策略。

使用方式：
    engine = StrategyEngine(event_engine, exec_engine, risk_manager)
    engine.add_strategy(MyStrategy, name="s1", symbols=["BTCUSDT"], params={})
    engine.start_all()
    engine.stop_all()
"""

from core.constants import EventType
from data_layer.logger import logger


class StrategyEngine:

    def __init__(self, event_engine, exec_engine, risk_manager=None):
        self.event_engine  = event_engine
        self.exec_engine   = exec_engine
        self.risk_manager  = risk_manager
        self.strategies    = {}   # {name: strategy_instance}

    def add_strategy(self, strategy_class, name: str, symbols: list, params: dict = None):
        """实例化并注册一个策略。"""
        instance = strategy_class(
            engine  = self.exec_engine,
            symbols = symbols,
            params  = params or {},
        )
        self.strategies[name] = instance

        # 注册事件处理器
        self.event_engine.register(EventType.TICK,       instance.on_tick)
        self.event_engine.register(EventType.BAR,        instance.on_bar)
        self.event_engine.register(EventType.ORDER_BOOK, instance.on_order_book)
        self.event_engine.register(EventType.FILL,       instance.on_fill)
        self.event_engine.register(EventType.ORDER,      instance.on_order_update)

        logger.info(f"[StrategyEngine] 策略已注册: {name}")
        return instance

    def start_all(self):
        for name, strategy in self.strategies.items():
            strategy.on_start()
            logger.info(f"[StrategyEngine] 策略已启动: {name}")

    def stop_all(self):
        for name, strategy in self.strategies.items():
            strategy.on_stop()
            logger.info(f"[StrategyEngine] 策略已停止: {name}")

"""
StrategyBase - 策略抽象基类（模板方法模式）

所有策略继承此类，按需覆盖生命周期钩子。
执行引擎通过 engine.send_order() 发送订单。
"""


class StrategyBase:

    def __init__(self, engine, name: str, symbols: list, params: dict):
        self.engine  = engine
        self.name    = name
        self.symbols = symbols
        self.params  = params

    # ─── 生命周期钩子（子类按需覆盖）────────────────────────────────────────────

    def on_start(self):
        pass

    def on_stop(self):
        pass

    def on_tick(self, event):
        pass

    def on_bar(self, event):
        pass

    def on_order_book(self, event):
        pass

    def on_fill(self, event):
        pass

    def on_order_update(self, event):
        pass

    # ─── 下单快捷方法 ─────────────────────────────────────────────────────────

    def market_buy(self, symbol: str, qty: float, reduce_only: bool = False):
        return self.engine.send_order(
            symbol=symbol, side="BUY", qty=qty,
            order_type="MARKET", reduce_only=reduce_only,
        )

    def market_sell(self, symbol: str, qty: float, reduce_only: bool = False):
        return self.engine.send_order(
            symbol=symbol, side="SELL", qty=qty,
            order_type="MARKET", reduce_only=reduce_only,
        )

    def limit_buy(self, symbol: str, qty: float, price: float):
        return self.engine.send_order(
            symbol=symbol, side="BUY", qty=qty,
            order_type="LIMIT", price=price, reduce_only=False,
        )

    def limit_sell(self, symbol: str, qty: float, price: float):
        return self.engine.send_order(
            symbol=symbol, side="SELL", qty=qty,
            order_type="LIMIT", price=price, reduce_only=False,
        )

    def cancel_order(self, symbol: str, order_id: int):
        return self.engine.cancel_order(symbol=symbol, order_id=order_id)

    def stop_order(self, symbol: str, side: str, order_type: str, stop_price: float):
        """下交易所侧条件单（STOP_MARKET / TAKE_PROFIT_MARKET），closePosition=True 关闭整仓。"""
        return self.engine.send_order(
            symbol=symbol, side=side, qty=0,
            order_type=order_type, stop_price=stop_price,
        )

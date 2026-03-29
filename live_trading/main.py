"""
系统入口（main.py）

职责：
    按照依赖顺序完成所有模块的初始化，并启动系统的三个主要运行体：
    1. EventEngine 后台线程（事件消费循环）
    2. WebSocket 行情线程（每个品种一个）
    3. MonitorEngine 定时线程（持仓状态打印）

    同时注册 SIGINT（Ctrl+C）和 SIGTERM 的优雅退出处理，
    确保系统关闭时按正确顺序释放资源（监控 → 事件引擎 → 网关）。

完整事件流（从行情到成交落库的全链路）：
    BinanceWebSocketClient（每个品种独立线程）
        │  TICK{symbol, price, qty, is_buyer_maker}
        ▼
    EventEngine（单队列，单线程串行处理）
        │
        ├─ TICK       → PositionManager.on_tick        更新各品种浮盈亏
        ├─ TICK       → MyStrategy.on_tick      Hawkes / Impact / OFI 更新
        ├─ ORDER_BOOK → MyStrategy.on_order_book 13 维特征 → AdaGrad → Platt
        │                   │ P(up)>0.62 or P(up)<0.38 时，edge 过滤后挂 Maker 限价单
        │                   ▼ send_order() 直接走 gateway（绕过 SIGNAL→ORDER 链路）
        │
        │  （BinanceGateway 从 User Data Stream 接收成交回报后发出 FILL 事件）
        ├─ FILL       → MyStrategy.on_fill      更新持仓/开仓价/冷却计数
        ├─ FILL       → PositionManager.on_fill        更新持仓数量和入场均价
        ├─ FILL       → RiskManager.on_fill_event      更新当日已实现 PnL
        ├─ FILL       → Database.on_fill_event         成交记录写入 SQLite
        │
        └─ ACCOUNT    → AccountManager.on_account_event 同步余额快照

定时任务（独立线程，不经过事件队列）：
    MonitorEngine: 每 30 秒打印一次持仓状态快照

初始化顺序原则：
    先创建基础设施（EventEngine、Gateway），再创建依赖它们的模块（Portfolio 层），
    最后创建最顶层的策略和监控，确保每个模块注册事件时引擎已就绪。
"""

import threading
import signal
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config, API_KEY, API_SECRET, SYMBOLS
from core.event_engine import EventEngine
from core.constants import EventType

from exchange.binance_gateway import BinanceGateway
from data_layer.websocket_client import BinanceWebSocketClient
from data_layer.order_book_live import MsOrderBookEngine

from portfolio.position_manager import PositionManager
from portfolio.account_manager import AccountManager
from portfolio.risk_manager import RiskManager
from execution.order_manager import OrderManager

from strategy.strategy_engine import StrategyEngine
from monitor.monitor_engine import MonitorEngine

from strategy.hft_lob_strategy import HFTLOBStrategy
STRATEGY_CLASS = HFTLOBStrategy
from storage.database import Database


def main():
    """
    系统主函数：按顺序初始化所有模块并启动系统。

    步骤：
        0. 配置校验（API Key 和风控参数有效性检查）
        1. 核心引擎（EventEngine + BinanceGateway）
        2. 组合层（PositionManager / AccountManager / RiskManager / OrderManager）
        3. 策略层（每个品种独立实例化 MyStrategy）
        4. 数据存储（若 config.SAVE_TRADES_TO_DB 开启）
        5. 监控引擎（MonitorEngine）
        6. 行情 WebSocket（每个品种独立线程）
        7. 启动所有组件
        8. 注册优雅退出信号并阻塞主线程
    """

    # ────────────────────────────────────────────────────────────────
    # 0. 配置校验
    #    在做任何网络连接前先验证配置，避免因配置错误在运行中途崩溃
    # ────────────────────────────────────────────────────────────────
    if not config.validate():
        print("配置校验失败，请检查 API Key 和风控参数")
        sys.exit(1)

    # ────────────────────────────────────────────────────────────────
    # 1. 核心引擎初始化
    #    EventEngine：系统的消息总线，必须最先创建，因为后续所有模块都依赖它
    #    BinanceGateway：建立 User Data Stream，开始接收成交和余额推送
    # ────────────────────────────────────────────────────────────────
    event_engine = EventEngine()

    gateway = BinanceGateway(
        API_KEY,
        API_SECRET,
        event_engine,
    )

    # ────────────────────────────────────────────────────────────────
    # 2. 组合层初始化
    #    每个模块在 __init__ 中自注册事件监听，无需手动 register：
    #    - PositionManager : 自注册 TICK（更新浮盈）/ FILL（更新持仓）
    #    - AccountManager  : 自注册 ACCOUNT（同步余额）
    #    - RiskManager     : 自注册 SIGNAL（风控校验）/ FILL（更新日盈亏）
    #    - OrderManager    : 自注册 ORDER（执行下单）
    #
    #    注意依赖关系：RiskManager 依赖 PositionManager（查询当前仓位），
    #    OrderManager 依赖 gateway（执行下单），因此必须按此顺序创建。
    # ────────────────────────────────────────────────────────────────
    position_manager = PositionManager(event_engine)
    account_manager = AccountManager(event_engine)
    risk_manager = RiskManager(position_manager, event_engine)
    order_manager = OrderManager(gateway, event_engine)

    # ────────────────────────────────────────────────────────────────
    # 3. 策略层初始化（StrategyEngine 统一管理生命周期和下单路由）
    #    StrategyEngine 自动注册 TICK / BAR / ORDER_BOOK / FILL 事件，
    #    每个品种一个策略实例，由 add_strategy() 触发 on_start() 钩子。
    # ────────────────────────────────────────────────────────────────
    strategy_engine = StrategyEngine(event_engine, gateway, risk_manager)

    if STRATEGY_CLASS is None:
        raise RuntimeError("请先在 main.py 顶部设置 STRATEGY_CLASS")

    strategy_cfg = config.STRATEGY_CONFIG
    for symbol in SYMBOLS:
        strategy_engine.add_strategy(
            STRATEGY_CLASS,
            name=f"strategy_{symbol}",
            symbols=[symbol],
            params=strategy_cfg,
        )

    # ────────────────────────────────────────────────────────────────
    # 4. 数据存储初始化
    #    按配置开关决定是否启用数据库记录，测试时可关闭节省 IO
    # ────────────────────────────────────────────────────────────────
    db = None
    if config.SAVE_TRADES_TO_DB:
        db = Database(config.DATABASE_PATH)
        # 注册 FILL 事件：每次成交后将记录写入 SQLite
        event_engine.register(EventType.FILL, db.on_fill_event)

    # ────────────────────────────────────────────────────────────────
    # 5. 监控引擎初始化
    #    MonitorEngine 持有 PositionManager 引用，定时读取持仓并打印
    #    interval=30：每 30 秒打印一次持仓快照
    # ────────────────────────────────────────────────────────────────
    monitor = MonitorEngine(position_manager, interval=30)

    # ────────────────────────────────────────────────────────────────
    # 6. 行情 WebSocket 启动
    #    每个品种两路数据流：
    #    - BinanceWebSocketClient (@trade) : 逐笔成交 → TICK 事件
    #    - MsOrderBookEngine (@depth)      : L2 前十档订单簿 → ORDER_BOOK 事件
    # ────────────────────────────────────────────────────────────────
    ws_clients = []
    ob_engines = []
    for symbol in SYMBOLS:
        ws = BinanceWebSocketClient(symbol, event_engine)
        ws_clients.append(ws)
        threading.Thread(target=ws.start, daemon=True).start()

        ob = MsOrderBookEngine(symbol, event_engine, top_n=10)
        ob_engines.append(ob)

    # ────────────────────────────────────────────────────────────────
    # 7. 启动系统组件
    #    EventEngine：启动后台消费线程，开始处理事件队列
    #    MonitorEngine：启动定时打印线程
    # ────────────────────────────────────────────────────────────────
    event_engine.start()
    monitor.start()

    # 打印系统启动概况，方便运维人员确认配置是否正确
    print("=" * 50)
    print("系统已启动")
    print(f"  交易品种: {SYMBOLS}")
    print(f"  策略数量: {len(strategy_engine.strategies)}")
    print(f"  L2 档位 : 前10档（@depth 实时流）")
    print(f"  数据存储: {'开启' if config.SAVE_TRADES_TO_DB else '关闭'}")
    print("=" * 50)

    # ────────────────────────────────────────────────────────────────
    # 8. 优雅退出处理
    #    注册 SIGINT（Ctrl+C）和 SIGTERM（kill 信号）的处理函数
    #    按反向依赖顺序关闭：监控 → 事件引擎 → 网关（先停上层，再停底层）
    # ────────────────────────────────────────────────────────────────
    def shutdown(sig, frame):
        """
        优雅关闭系统。

        关闭顺序：
            1. MonitorEngine：停止定时器，不再打印新的状态
            2. EventEngine：停止事件消费循环（等待当前处理完成）
            3. BinanceGateway：关闭 WebSocket，停止接收交易所推送
        """
        print("\n正在关闭系统...")
        monitor.stop()                          # 停止监控定时器
        strategy_engine.stop_all()              # 触发所有策略 on_stop()（撤单/平仓）
        for _ws in ws_clients:
            _ws.stop()                          # 停止 TICK 重连循环
        for _ob in ob_engines:
            _ob.close()                         # 关闭 L2 订单簿 WebSocket
        event_engine.stop()                     # 停止事件消费线程
        gateway.close()                         # 关闭网关 WebSocket 连接
        if db is not None:
            db.close()                          # 刷写剩余成交记录并关闭 SQLite
        print("系统已关闭")
        sys.exit(0)

    # 注册两种退出信号：Ctrl+C 发送 SIGINT，Docker/systemd 停止发送 SIGTERM
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 阻塞主线程，等待信号触发退出
    # signal.pause() 使主线程进入休眠直到收到任意信号，避免 while True 的 CPU 空转
    signal.pause()


if __name__ == "__main__":
    main()

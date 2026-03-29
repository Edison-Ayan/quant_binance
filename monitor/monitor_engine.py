"""
监控引擎（MonitorEngine）

职责：
    定时打印当前所有持仓的状态快照（数量、入场均价、浮动盈亏），
    为运维人员提供实时的账户状况可视化。

设计思路：
    使用 threading.Timer 而不是 time.sleep() 循环的原因：
    - sleep() 循环会永久阻塞一个线程，且停止时不够干净
    - Timer 是一次性定时器，触发后自动结束。通过在回调中重新创建 Timer，
      实现"自我调度"的周期性执行
    - stop() 方法可以通过取消 Timer 立即停止，非常干净，无需等待 sleep 结束

    Timer 线程设为 daemon=True，确保主程序退出时自动销毁，
    无需显式 join，适合不影响主流程生命周期的辅助任务。

在系统中的位置：
    main.py 启动后独立运行，不接入事件引擎，
    直接读取 PositionManager.positions 字典（只读，无需加锁）。
"""

import threading


class MonitorEngine:
    """
    持仓状态定时监控引擎。

    每隔指定时间打印一次当前所有持仓的快照，方便实时监控账户状况。
    """

    def __init__(self, position_manager, interval: int = 30):
        """
        初始化监控引擎。

        参数：
            position_manager : PositionManager 实例，从中读取持仓状态
            interval (int)   : 打印间隔（秒），默认 30 秒
                               间隔越短，监控越实时，但终端输出越密集
        """
        # 持仓管理器引用，只读方式访问 positions 字典
        self.position_manager = position_manager

        # 定时打印间隔（秒）
        self.interval = interval

        # 当前 Timer 对象引用，用于 stop() 时取消定时器
        self._timer: threading.Timer = None

    # -------------------------
    # 生命周期管理
    # -------------------------

    def start(self):
        """
        启动定时监控。

        立即执行一次打印（_schedule 内部先调用 print_status），
        然后设定下次触发时间。这样启动后不必等待一个完整的 interval 就能看到状态。
        """
        self._schedule()

    def stop(self):
        """
        停止定时监控。

        取消尚未触发的 Timer，确保在系统关闭时不再有新的打印输出，
        也避免因 Timer 持有 position_manager 引用而阻止垃圾回收。
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    # -------------------------
    # 自我调度逻辑
    # -------------------------

    def _schedule(self):
        """
        执行一次状态打印，然后安排下一次调度。

        自我调度模式：
            _schedule() 先执行业务逻辑（print_status），
            再创建一个新的 Timer 指向自身（self._schedule），
            从而实现周期性循环，直到 stop() 被调用取消 Timer。

        为什么不用 while + sleep 循环？
            Timer 方式更灵活：stop() 可以立即取消而不必等待 sleep 结束，
            且不会阻塞专用线程（Timer 用完即释放，下次重新创建）。
        """
        # 先执行状态打印
        self.print_status()

        # 创建新的一次性 Timer，interval 秒后再次调用 _schedule
        self._timer = threading.Timer(self.interval, self._schedule)

        # 设为守护线程：主程序退出时 Timer 自动销毁，无需手动清理
        self._timer.daemon = True

        # 启动定时器（非阻塞，Timer 在后台独立线程中等待）
        self._timer.start()

    # -------------------------
    # 状态打印
    # -------------------------

    def print_status(self):
        """
        打印当前所有持仓的状态快照。

        格式化输出各品种的持仓数量（qty）、入场均价（entry）和浮动盈亏（pnl），
        方便运维人员快速了解账户整体状况。

        跳过 qty == 0 的品种（曾有过成交但已完全平仓，无需显示）。
        """
        # 读取当前全部持仓（只读访问，事件引擎单线程写入，无并发冲突）
        positions = self.position_manager.positions

        if not positions:
            print("[Monitor] 当前无持仓")
            return

        print("-" * 40)
        print(f"[Monitor] 持仓状态（共 {len(positions)} 个品种）")
        print("-" * 40)

        for symbol, pos in positions.items():

            # 跳过已平仓品种（qty == 0 表示无持仓，无需显示）
            if pos.qty == 0:
                continue

            # 格式化输出：品种名、持仓量、入场均价、浮动盈亏
            # pnl 前加 + 号（{:+.4f}），正负一目了然
            print(
                f"  {symbol:<12} "          # 品种名，左对齐 12 字符
                f"qty={pos.qty:>10.4f}  "   # 持仓量，右对齐 10 字符，4 位小数
                f"entry={pos.entry_price:>12.4f}  "   # 入场均价
                f"pnl={pos.unrealized_pnl:>+12.4f}"  # 浮盈亏（带正负号）
            )

        print("-" * 40)

"""
事件驱动引擎（EventEngine）

职责：
    本模块是整个量化系统的"神经中枢"，负责在所有业务模块之间传递事件消息。
    所有模块（行情、策略、风控、执行、持仓）均不直接相互调用，而是通过向
    EventEngine 投递事件（put）和注册处理函数（register）来解耦通信。

设计要点：
    - 单队列（Queue）：线程安全，所有事件的唯一入口
    - 单线程消费：`_run` 方法在独立后台线程中串行处理，避免并发竞争
    - 超时等待（timeout=1s）：防止 `queue.get()` 永久阻塞，使 `_active=False`
      时可以及时退出循环
    - 异常隔离：每个 handler 的异常独立捕获，一个 handler 崩溃不影响其他处理逻辑

事件流（完整链路）：
    TICK → Strategy.on_tick → SIGNAL → RiskManager.on_signal_event
         → ORDER → OrderManager.on_order_event → (网关下单)
         → FILL → PositionManager / RiskManager / Database
         → ACCOUNT → AccountManager
"""

from queue import Queue, Empty
from typing import Callable, Dict, List
import threading

from core.event import Event
from core.constants import EventType


class EventEngine:
    """
    单队列、单线程的事件总线。

    所有模块通过 `put()` 发布事件，通过 `register()` 订阅特定类型事件。
    消费者 handler 在同一个后台线程中被串行调用，无需考虑并发竞争。
    """

    def __init__(self):

        # 事件队列：线程安全的 FIFO 队列，所有生产者都向此队列放入事件
        self.queue = Queue()

        # 事件处理器字典：key 为 EventType，value 为该类型的所有 handler 列表
        # 一个事件类型可以有多个订阅者（例如 FILL 同时被持仓、风控、数据库处理）
        self.handlers: Dict[EventType, List[Callable]] = {}

        # 控制标志：False 表示引擎已停止，_run 循环应退出
        self._active = False

        # 消费者线程：独立后台线程，持续从队列取事件并分发给 handler
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        """
        事件消费循环（在独立线程中运行）。

        设计原则：
        1. 使用 timeout=1 的阻塞获取，而不是 get_nowait()，
           这样在队列为空时 CPU 不会空转（自旋等待），节省资源
        2. 超时后检查 _active 标志，确保 stop() 后线程能正常退出
        3. 每个 handler 单独 try/except，保证一个 handler 报错不会
           导致同一事件的其他 handler 被跳过
        """
        handlers = self.handlers   # 局部引用，避免每次循环属性查找
        while self._active:
            try:
                # 50ms 超时：比原来的 1s 快 20 倍，stop() 响应更及时
                event = self.queue.get(timeout=0.05)
            except Empty:
                continue

            event_handlers = handlers.get(event.type)
            if event_handlers:
                for handler in event_handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        print(f"Handler error [{event.type}]: {e}")

    def register(self, event_type: EventType, handler: Callable):
        """
        注册事件处理函数。

        同一事件类型可注册多个 handler（追加而非覆盖）。
        通常在各模块的 __init__ 中调用，实现"订阅"语义。

        参数：
            event_type (EventType) : 要订阅的事件类型
            handler    (Callable)  : 事件触发时要调用的函数，签名为 handler(event: Event)
        """
        # 若该事件类型尚无任何订阅者，先创建空列表
        if event_type not in self.handlers:

            self.handlers[event_type] = []

        # 追加 handler，不覆盖已有订阅者
        self.handlers[event_type].append(handler)

    def put(self, event: Event):
        """
        向事件队列投递事件（生产者调用）。

        此方法线程安全，可从任何线程调用（行情线程、网关回调线程等）。
        事件被放入队列后，由 `_run` 中的消费者线程异步处理。

        参数：
            event (Event) : 要投递的事件对象
        """
        self.queue.put(event)

    def start(self):
        """
        启动事件引擎。

        将消费者线程设为 daemon=True，确保主程序退出时后台线程自动终止，
        无需手动 join（适合长期运行的量化系统）。
        """
        self._active = True
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """
        停止事件引擎。

        先将 _active 设为 False，让 _run 循环在下次超时后自然退出，
        再 join 等待线程实际结束（最多等待 5 秒，防止无限阻塞）。
        """
        self._active = False
        if self._thread.is_alive():
            self._thread.join(timeout=1)

"""
日志模块（QuantLogger）

职责：
    为整个量化系统提供统一的日志记录接口，支持控制台实时输出和按日期文件归档。
    封装 Python 标准 logging 库，对外提供语义化的方法（info / debug / trade 等），
    隐藏 logging 库的配置细节。

设计要点：
    1. 全局单例：模块底部创建 `logger = QuantLogger()` 单例，其他模块只需
       `from utils.logger import logger` 即可使用，无需自己初始化
    2. 双输出：同时写入控制台（实时监控）和日志文件（持久归档）
    3. 幂等初始化：检查 `not self.logger.handlers` 防止模块被多次 import 时
       重复添加 handler，避免日志重复打印
    4. 按日期命名文件：logs/quant_YYYYMMDD.log，每天产生一个文件，便于按日查阅
    5. 成交专用日志：trade() 方法使用固定格式，便于从日志中提取成交流水
"""

import logging
import os
from datetime import datetime

from config.settings import config


class QuantLogger:
    """
    量化交易专用日志记录器。

    封装 Python 标准 logging.Logger，提供语义化接口并自动配置
    控制台和文件双输出 handler。
    """

    def __init__(self, name: str = "quant_trading"):
        """
        初始化日志记录器，配置控制台和文件输出。

        参数：
            name (str) : logger 名称，同名 logger 共享同一实例（Python logging 机制）
                         默认 "quant_trading"，确保全局唯一
        """
        # 获取（或创建）名为 name 的 logger 实例
        # Python logging 使用名称池，相同名称始终返回同一实例
        self.logger = logging.getLogger(name)

        # 从配置文件读取日志级别（DEBUG / INFO / WARNING / ERROR）
        # 使用 getattr 安全转换，若配置值无效则回退到 INFO
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

        # 幂等检查：若 handler 已存在（模块被重复 import），跳过初始化
        # 避免同一条日志被打印多次（每个 handler 打印一次）
        # 禁止传播到 root logger，防止日志被 root handler 重复打印
        self.logger.propagate = False
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """
        配置控制台（StreamHandler）和文件（FileHandler）两个输出 handler。

        控制台 handler：日志级别与全局配置一致，DEBUG 消息可按需打开
        文件 handler：始终记录 INFO 及以上级别，保留关键操作记录，
                      避免 DEBUG 消息填满磁盘
        """
        # 统一的日志格式：时间 - 模块名 - 级别 - 消息
        # 例如：2024-01-15 10:30:00,123 - quant_trading - INFO - 系统已启动
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # ── 控制台输出（实时监控）──────────────────────────────────────
        console_handler = logging.StreamHandler()
        # 控制台级别与全局配置一致（开发时可设 DEBUG，生产时设 INFO）
        console_handler.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # ── 文件输出（持久归档）───────────────────────────────────────
        # 若 logs 目录不存在则创建（exist_ok=True 避免并发创建时报错）
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # 按日期命名日志文件，例如：logs/quant_20240115.log
        # 每天产生独立文件，便于按日期归档和清理
        log_file = os.path.join(
            log_dir, f"quant_{datetime.now().strftime('%Y%m%d')}.log"
        )
        # encoding="utf-8"：支持中文日志内容，避免 Windows 默认编码（GBK）乱码
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        # 文件固定记录 INFO 及以上，不记录 DEBUG 调试信息（保持文件简洁）
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    # -------------------------
    # 日志接口（语义化封装）
    # -------------------------

    def info(self, message: str):
        """记录 INFO 级别日志，用于系统正常运行的关键节点记录。"""
        self.logger.info(message)

    def debug(self, message: str):
        """记录 DEBUG 级别日志，用于开发调试，生产环境通常不输出。"""
        self.logger.debug(message)

    def warning(self, message: str):
        """记录 WARNING 级别日志，用于非致命异常或数据质量问题提醒。"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录 ERROR 级别日志，用于可恢复错误（如 API 调用失败）。"""
        self.logger.error(message)

    def critical(self, message: str):
        """记录 CRITICAL 级别日志，用于致命错误（如数据库损坏、资金异常）。"""
        self.logger.critical(message)

    def trade(self, symbol: str, side: str, price: float, size: float):
        """
        成交记录专用日志，格式固定便于后期数据分析和日志解析。

        固定格式：TRADE | {symbol} {side} {size}@{price}
        例如：TRADE | BTCUSDT BUY 0.01@45000.0

        参数：
            symbol (str)   : 交易对
            side   (str)   : "BUY" 或 "SELL"
            price  (float) : 成交价格
            size   (float) : 成交数量
        """
        # 使用 INFO 级别，确保成交日志始终被记录到文件（不受 DEBUG 开关影响）
        self.info(f"TRADE | {symbol} {side} {size}@{price}")


# ─── 全局单例 ────────────────────────────────────────────────────────────────
# 在模块加载时创建单例，所有其他模块通过以下方式直接使用：
#   from utils.logger import logger
#   logger.info("消息内容")
#
# 使用单例而不是让每个模块自己初始化的原因：
# 1. 避免重复配置
# 2. 所有模块的日志输出到同一个文件
# 3. 可以统一修改日志配置而无需修改每个模块
logger = QuantLogger()

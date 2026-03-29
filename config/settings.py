"""
系统配置文件（settings.py）

职责：
    集中管理整个量化交易系统的所有可调参数，支持通过环境变量覆盖默认值，
    避免将敏感信息（API Key）硬编码在代码中。

配置加载优先级：
    环境变量 > 代码默认值
    例如：os.getenv("BINANCE_API_KEY", "YOUR_API_KEY")
    - 若环境变量 BINANCE_API_KEY 已设置，使用环境变量的值
    - 否则使用默认占位值 "YOUR_API_KEY"

安全注意事项：
    - API Key 和 Secret 绝对不要硬编码到代码中
    - 生产环境通过 export BINANCE_API_KEY=xxx 设置环境变量，或使用 .env 文件
    - 本文件不应提交到版本控制（若含有真实 Key）

向后兼容性：
    模块底部将 config 对象的常用属性导出为模块级变量（API_KEY、SYMBOL 等），
    使 `from config.settings import TRADE_SIZE` 这种旧式 import 仍然有效。
"""

import os


class Config:
    """
    系统配置管理类。

    集中管理所有模块所需的配置参数，通过 os.getenv 支持环境变量覆盖，
    并提供配置校验和按策略获取品种列表等工具方法。
    """

    def __init__(self):
        # ── API 配置 ─────────────────────────────────────────────────
        # 从环境变量读取 API Key，生产环境必须设置，否则 validate() 会返回 False
        self.API_KEY    = os.getenv("BINANCE_API_KEY",    "")
        self.API_SECRET = os.getenv("BINANCE_API_SECRET", "")

        # Testnet API Key（测试网，不要用实盘 Key！）
        self.TESTNET_API_KEY    = os.getenv("TESTNET_API_KEY",    "vmvYBh2pZEOpZSTKzUY2ikylUdPqdJrfuDS6NaVUIj3UyA6a0PdQlMjRFHsQo1nv")
        self.TESTNET_API_SECRET = os.getenv("TESTNET_API_SECRET", "g42XYG18HhdlUl9InwDRu1ETz1VCQVaIm86XXhMGOeEX9QQzqJDhQtNwPm4WDyan")

        # ── 交易品种配置 ──────────────────────────────────────────────
        # 单品种（兼容旧代码和测试用）
        self.SYMBOL = "ETHUSDT"

        # 多品种列表（主程序遍历此列表启动 WebSocket 和策略）
        self.SYMBOLS = [
            "ETHUSDT",  # 以太坊/USDT 合约
            "BTCUSDT",  # 比特币/USDT 合约
            "ADAUSDT",  # ADA/USDT 合约
            "DOTUSDT"   # DOT/USDT 合约
        ]

        # ── 风险控制参数 ──────────────────────────────────────────────
        # 每笔交易的基础下单量（合约张数或币数），供 PositionSizer 使用
        self.TRADE_SIZE = 0.01

        # 账户杠杆倍数，建议不超过 10 倍（Binance 合约最高 125 倍，但风险极高）
        self.LEVERAGE = 5

        # 最大持仓比例（相对于账户总净值，当前为占位参数，风控逻辑在 RiskManager 中）
        self.MAX_POSITION_SIZE = 0.1   # 单品种最大仓位不超过账户的 10%

        # 最大回撤限制（相对于账户净值，触发后应停止交易）
        self.MAX_DRAWDOWN = 0.05       # 最大允许 5% 的回撤

        # ── 系统基础配置 ──────────────────────────────────────────────
        # SQLite 数据库文件路径（相对于运行目录）
        self.DATABASE_PATH = "quant_trades.db"

        # 日志级别：DEBUG（详细）/ INFO（常规）/ WARNING（仅警告）/ ERROR（仅错误）
        self.LOG_LEVEL = "INFO"

        # WebSocket 连接超时时间（秒），超过此时间未收到消息则触发重连
        self.WEBSOCKET_TIMEOUT = 30

        # ── 性能监控开关 ──────────────────────────────────────────────
        # 是否启用性能监控（当前为占位符，未来可接入 Prometheus 等监控系统）
        self.PERFORMANCE_MONITORING = True

        # 是否将成交记录写入 SQLite 数据库（可在测试时关闭节省 IO）
        self.SAVE_TRADES_TO_DB = True

    def validate(self) -> bool:
        """
        校验关键配置是否已正确设置。

        在系统启动时调用，若配置不合法则提前报错，
        避免因配置错误导致运行中途出现不明原因的 API 拒绝。

        校验项目：
            1. API Key 是否仍为占位值（未被真实 Key 替换）
            2. 杠杆倍数是否在安全范围内（> 20 倍风险极高）

        返回：
            bool : True=配置有效，False=配置存在问题（详情见打印提示）
        """
        # 检查 API Key 是否还是占位符（未设置真实的 Key）
        if not self.API_KEY or not self.API_SECRET:
            print("警告: 请通过环境变量设置 API 密钥：\n"
                  "  export BINANCE_API_KEY=your_key\n"
                  "  export BINANCE_API_SECRET=your_secret")
            return False

        # 高杠杆警告：超过 20 倍杠杆在剧烈行情下极易爆仓
        if self.LEVERAGE > 20:
            print("警告: 杠杆过高，建议降低杠杆")
            return False

        return True


# ─── 全局配置单例 ─────────────────────────────────────────────────────────────
# 在模块加载时创建全局配置实例，整个系统共享同一个 Config 对象
# 其他模块通过 `from config.settings import config` 获取全局配置
config = Config()

# ─── 向后兼容的模块级变量 ────────────────────────────────────────────────────
# 保持旧代码（如 from config.settings import TRADE_SIZE）的兼容性
# 新代码应优先使用 config.TRADE_SIZE 等属性访问方式
API_KEY = config.API_KEY
API_SECRET = config.API_SECRET
SYMBOL = config.SYMBOL
SYMBOLS = config.SYMBOLS
TRADE_SIZE = config.TRADE_SIZE
LEVERAGE = config.LEVERAGE
DATABASE_PATH = config.DATABASE_PATH
LOG_LEVEL = config.LOG_LEVEL

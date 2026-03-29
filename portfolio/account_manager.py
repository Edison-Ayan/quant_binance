"""
账户余额管理模块（AccountManager）

职责：
    同步并维护本地账户余额快照，供其他模块（如风控、仓位计算）查询可用资金。

在系统事件流中的位置：
    BinanceGateway 接收 Binance ACCOUNT_UPDATE 推送
        │  发出 ACCOUNT{balances, positions}
        ▼
    AccountManager.on_account_event() → 更新本地余额缓存

为什么需要本地余额缓存？
    频繁调用 REST API 查询账户余额会消耗 API 请求配额（Rate Limit），
    且有网络延迟。通过订阅 ACCOUNT_UPDATE 推送，每次账户变动时自动同步，
    查询时直接读取本地缓存，无需网络请求，延迟极低。

Binance ACCOUNT_UPDATE 字段说明：
    balances 列表每项格式：
        {
            "a":  "USDT",     # asset 资产名称
            "wb": "1000.00",  # walletBalance 钱包余额（含未实现盈亏）
            "cw": "990.00"    # crossWalletBalance 全仓保证金余额（扣除浮亏后）
        }
    本模块存储 walletBalance（wb），即名义余额，
    若需要可用保证金则应使用 crossWalletBalance（cw）。
"""

from core.constants import EventType


class AccountManager:
    """
    账户余额管理器。

    监听 ACCOUNT 事件，将交易所推送的最新余额快照同步到本地字典，
    对外提供 O(1) 的余额查询接口。
    """

    def __init__(self, event_engine=None):
        """
        初始化账户管理器。

        参数：
            event_engine (EventEngine) : 全局事件引擎，传 None 时不注册事件监听
                                         （主要用于单元测试场景，可手动调用 update_balance）
        """
        # 余额字典：key=资产名称（如 "USDT"），value=钱包余额（float）
        # 初始为空，等待首次 ACCOUNT_UPDATE 推送填充
        self.balances: dict = {}

        # 仅在传入 event_engine 时注册监听，支持无事件引擎的测试场景
        if event_engine is not None:
            event_engine.register(EventType.ACCOUNT, self.on_account_event)

    # -------------------------
    # 事件处理
    # -------------------------

    def on_account_event(self, event):
        """
        接收 ACCOUNT 事件，同步交易所余额快照到本地缓存。

        Binance ACCOUNT_UPDATE 中 balances 字段格式：
            [{"a": "USDT", "wb": "1000.00", "cw": "990.00"}, ...]
            a  = asset：资产名称
            wb = walletBalance：钱包余额（含未实现盈亏的名义余额）
            cw = crossWalletBalance：全仓余额（扣除浮亏后的可用保证金）

        注意：ACCOUNT_UPDATE 只推送"有变化的资产"，
        未发生变化的资产不会出现在 balances 列表中，因此使用更新而非全量替换。

        参数：
            event (Event) : EventType.ACCOUNT 事件
        """
        # 遍历本次推送中有余额变化的所有资产
        for b in event.data.get("balances", []):
            # "a"：资产名称（asset），如 "USDT"、"BTC"
            asset = b.get("a", "")
            # "wb"：钱包余额（walletBalance），字符串格式需转 float
            wallet_balance = float(b.get("wb", 0))
            # 只更新有效资产（asset 非空），跳过格式异常的条目
            if asset:
                self.update_balance(asset, wallet_balance)

    # -------------------------
    # 查询 / 更新接口
    # -------------------------

    def update_balance(self, asset: str, balance: float):
        """
        手动更新指定资产的余额（也供 on_account_event 调用）。

        参数：
            asset   (str)   : 资产名称，例如 "USDT"
            balance (float) : 最新余额
        """
        self.balances[asset] = balance

    def get_balance(self, asset: str) -> float:
        """
        查询指定资产的当前余额。

        参数：
            asset (str) : 资产名称，例如 "USDT"

        返回：
            float : 当前余额；若该资产从未收到更新则返回 0.0
        """
        return self.balances.get(asset, 0.0)

    def get_all_balances(self) -> dict:
        """
        返回所有资产余额的副本（dict 浅拷贝）。

        返回副本而非引用，防止调用方意外修改内部状态。

        返回：
            dict : {资产名称: 余额} 的完整字典副本
        """
        return dict(self.balances)

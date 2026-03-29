"""
Alpha Factory - Binance Testnet 测试入口

目的：
    用 Binance Futures Testnet 验证下单代码的正确性，
    无需真实资金，但 API 调用是真实的（能暴露精度错误、权限问题等）。

Testnet 申请：
    https://testnet.binancefuture.com
    → 注册后领取测试币（USDT）
    → 在 "API Management" 创建 API Key

Testnet 限制（与实盘的差异）：
    ✅ 支持所有合约下单接口
    ✅ 支持 WebSocket 行情（但行情可能不活跃）
    ⚠️  行情活跃度不如实盘（成交量极低，不适合测试信号）
    ⚠️  WebSocket 推送频率低
    ❌  不能测试真实滑点和流动性

最佳实践：
    行情数据用实盘（wss://fstream.binance.com）
    下单接口用 Testnet（https://testnet.binancefuture.com）
    → 两者地址不同，本脚本已分开配置

使用方式：
    # 设置 Testnet API Key（不要用实盘 Key！）
    set TESTNET_API_KEY=your_testnet_key
    set TESTNET_API_SECRET=your_testnet_secret

    python live_trading/run_alpha_factory_testnet.py
"""

import asyncio
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

from data_layer.logger import logger
from data_layer.multi_symbol_ws import MultiSymbolFeed
from data_layer.rest_fetcher import RestFetcher
from alpha_factory.universe_filter import UniverseFilter
from alpha_factory.alpha_strategy import AlphaFactoryStrategy
from storage.database import Database


# ── Testnet 配置 ──────────────────────────────────────────────────────────────

TESTNET_API_URL  = "https://testnet.binancefuture.com"  # REST
TESTNET_WS_URL   = "wss://stream.binancefuture.com"     # WebSocket（行情）

# 从 settings 读取（环境变量优先，回退到 settings.py 中的默认值）
from config.settings import config as _cfg
TESTNET_API_KEY    = _cfg.TESTNET_API_KEY
TESTNET_API_SECRET = _cfg.TESTNET_API_SECRET


# ── Testnet 执行引擎 ──────────────────────────────────────────────────────────

class _FuturesTestnetClient(Client):
    """
    覆盖 Client.ping()，避免连接现货 Testnet（testnet.binance.vision，国内常被拦截）。
    我们只用期货 API，初始化时改为 ping 期货 Testnet 以验证连通性。
    """
    def ping(self):
        return self.futures_ping()


class TestnetEngine:
    """
    连接 Binance Testnet 的执行引擎。

    与 PaperEngine 的区别：
    - 真实调用 Binance Testnet API 下单
    - 能暴露精度错误、权限问题、资金不足等真实问题
    - 资金是假的，不会有真实损失
    """

    def __init__(self, api_key: str, api_secret: str):
        self.client = _FuturesTestnetClient(
            api_key,
            api_secret,
            testnet=True,   # 关键：启用 Testnet 模式（期货走 testnet.binancefuture.com）
            requests_params={"timeout": 10},
        )
        self.client.REQUEST_TIMEOUT = 10
        # 允许本地时钟与服务器最多有 5 秒偏差（代理/时钟漂移场景）
        self.client.recvWindow = 5000
        # 自动同步本地时钟与服务器时钟偏差（解决 -1021 Timestamp 错误）
        self._sync_server_time()
        self._step_sizes: dict = {}      # {symbol: qty decimal_places}
        self._tick_sizes: dict = {}      # {symbol: price decimal_places}
        self.valid_symbols: set = set()  # Testnet 上实际存在的合约
        self._invalid_symbols: set = set()  # 下单时发现的无效合约（运行时黑名单）
        self._leverage_set: set = set()     # 已设置过杠杆的品种（避免重复调用）
        self._leverage: int = 10            # 目标杠杆倍数
        self._actual_leverage: dict = {}    # {symbol: 实际生效杠杆}（因品种限制可能低于目标）
        self._validate_connection()
        self._load_precision()
        self._set_leverage()

    def _sync_server_time(self):
        """
        获取 Binance 服务器时间并计算本地时钟偏差，注入客户端。

        解决 -1021 "Timestamp for this request was Xms ahead of server time" 错误。

        python-binance 在每次请求前执行：
            timestamp = int(time.time() * 1000) + self.timestamp_offset
        因此：
            - 本地时钟偏快 → 需要负偏移 → timestamp_offset = server_time - local_time（负值）
            - 本地时钟偏慢 → 需要正偏移 → timestamp_offset = server_time - local_time（正值）
        统一公式：timestamp_offset = server_time - local_time
        """
        try:
            server_time = self.client.get_server_time()["serverTime"]   # ms
            local_time  = int(time.time() * 1000)
            offset_ms   = server_time - local_time   # 负值 = 本地偏快，正值 = 本地偏慢
            self.client.timestamp_offset = offset_ms
            logger.info(f"[Testnet] 时钟同步完成 | timestamp_offset={offset_ms:+d}ms")
        except Exception as e:
            logger.warning(f"[Testnet] 时钟同步失败（将继续尝试连接）: {e}")

    def _validate_connection(self):
        """验证 Testnet 连接和账户状态"""
        try:
            account = self.client.futures_account()
            balance = float(account.get("totalWalletBalance", 0))
            logger.info(f"[Testnet] 连接成功 | Testnet 余额: {balance:.2f} USDT")
        except BinanceAPIException as e:
            logger.error(f"[Testnet] 连接失败: {e}")
            raise

    def _load_precision(self):
        """从交易所拉取每个品种的数量精度（stepSize）和价格精度（tickSize），存为小数位数"""
        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                sym = s["symbol"]
                self.valid_symbols.add(sym)
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        step = f["stepSize"].rstrip("0") or "1"
                        self._step_sizes[sym] = len(step.split(".")[1]) if "." in step else 0
                    elif f["filterType"] == "PRICE_FILTER":
                        tick = f["tickSize"].rstrip("0") or "1"
                        self._tick_sizes[sym] = len(tick.split(".")[1]) if "." in tick else 0
            logger.info(
                f"[Testnet] 精度规则已加载，共 {len(self._step_sizes)} 个品种 "
                f"(Testnet 有效合约: {len(self.valid_symbols)} 个)"
            )
        except Exception as e:
            logger.warning(f"[Testnet] 精度规则加载失败，使用默认3位: {e}")

    def _format_price(self, symbol: str, price: float) -> str:
        """按交易所 tickSize 格式化止盈止损价格"""
        decimals = self._tick_sizes.get(symbol, 6)
        return f"{price:.{decimals}f}"

    def _quantize_qty(self, symbol: str, qty: float) -> str:
        """按交易所 stepSize 截断数量，避免精度拒单"""
        decimals = self._step_sizes.get(symbol, 3)
        factor = 10 ** decimals
        truncated = int(qty * factor) / factor   # 向下截断
        return f"{truncated:.{decimals}f}"

    def _set_leverage(self, leverage: int = 20):
        """记录目标杠杆（真正的设置在首次开仓时按品种调用）"""
        self._leverage = leverage
        logger.info(f"[Testnet] 目标杠杆: {leverage}x（首次开仓时逐品种设置）")

    def _ensure_leverage(self, symbol: str):
        """首次对某品种下单前设置杠杆，后续跳过。
        若目标杠杆不被支持（-4028），自动降档到该币允许的最大值。
        """
        if symbol in self._leverage_set:
            return
        # 降档序列：20 → 10 → 5 → 3 → 2 → 1
        candidates = sorted(
            {lev for lev in [self._leverage, 10, 5, 3, 2, 1] if lev <= self._leverage},
            reverse=True,
        )
        for lev in candidates:
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=lev)
                self._leverage_set.add(symbol)
                self._actual_leverage[symbol] = lev
                if lev < self._leverage:
                    logger.warning(
                        f"[Testnet] {symbol} 最大杠杆为 {lev}x（目标{self._leverage}x不支持），已自动调低"
                    )
                else:
                    logger.info(f"[Testnet] {symbol} 杠杆已设为 {lev}x")
                return
            except BinanceAPIException as e:
                if e.code == -4028:
                    continue   # 该档不支持，继续降
                logger.warning(f"[Testnet] {symbol} 杠杆设置失败（继续下单）: {e}")
                return
        logger.warning(f"[Testnet] {symbol} 所有杠杆档位均失败，跳过杠杆设置")

    def send_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MARKET",
        price: float = None,
        reduce_only: bool = False,
        stop_price: float = None,
    ) -> dict:
        """
        向 Testnet 发送真实订单（但资金是假的）。

        order_type 支持：
            MARKET              普通市价单（需要 qty）
            LIMIT               限价单（需要 qty + price）
            STOP_MARKET         止损市价单（需要 stop_price，closePosition=True）
            TAKE_PROFIT_MARKET  止盈市价单（需要 stop_price，closePosition=True）
        """
        if symbol in self._invalid_symbols:
            logger.warning(f"[Testnet] 跳过 {side} {symbol}：已知无效合约")
            return None

        self._ensure_leverage(symbol)

        _STOP_TYPES = ("STOP_MARKET", "TAKE_PROFIT_MARKET")

        try:
            if order_type in _STOP_TYPES:
                # 条件单：由交易所触发，关闭整个仓位，不需要 quantity
                if not stop_price or stop_price <= 0:
                    logger.warning(f"[Testnet] {symbol} {order_type} 缺少 stop_price，跳过")
                    return None
                params = {
                    "symbol":        symbol,
                    "side":          side,
                    "type":          order_type,
                    "stopPrice":     self._format_price(symbol, stop_price),
                    "closePosition": "true",
                }
                result = self.client.futures_create_order(**params)
                logger.info(
                    f"[Testnet] {order_type} {side} {symbol} "
                    f"stopPrice={self._format_price(symbol, stop_price)}"
                    f" → orderId={result.get('orderId', 'N/A')}"
                )
                return result

            # 普通市价 / 限价单
            qty_str   = self._quantize_qty(symbol, qty)
            qty_final = float(qty_str)

            if qty_final <= 0:
                logger.warning(f"[Testnet] 跳过 {side} {symbol}：精度截断后数量为0")
                return None

            params = {
                "symbol":     symbol,
                "side":       side,
                "type":       order_type,
                "quantity":   qty_str,
                "reduceOnly": str(reduce_only).lower(),
            }
            if order_type == "LIMIT" and price:
                params["price"]       = f"{price:.4f}"
                params["timeInForce"] = "GTC"

            result = self.client.futures_create_order(**params)
            logger.info(
                f"[Testnet] {side} {qty_str} {symbol} → orderId={result['orderId']}"
                f" status={result['status']}"
            )
            return result

        except BinanceAPIException as e:
            if e.code in (-1121, -4411, -4140):  # 无效合约 / TradFi协议 / 合约状态异常
                self._invalid_symbols.add(symbol)
                self.valid_symbols.discard(symbol)
                logger.warning(f"[Testnet] {symbol} 不可交易（{e.code}），已加入黑名单，后续自动跳过")
            else:
                logger.error(f"[Testnet] 下单失败 {symbol} {side} {qty} stop={stop_price}: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: int):
        try:
            self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"[Testnet] 撤单 {symbol} #{order_id}")
        except BinanceAPIException as e:
            logger.error(f"[Testnet] 撤单失败: {e}")

    def get_positions(self) -> list:
        """查询当前 Testnet 持仓（代理断连时自动重试一次）"""
        for attempt in range(2):
            try:
                return [
                    p for p in self.client.futures_position_information()
                    if float(p["positionAmt"]) != 0
                ]
            except Exception as e:
                if attempt == 0:
                    time.sleep(2)   # 等待代理恢复后重试
                else:
                    logger.warning(f"[Testnet] 持仓查询失败（代理不稳定，不影响交易）: {type(e).__name__}")
        return []


# ── 主运行器 ──────────────────────────────────────────────────────────────────

class AlphaFactoryTestnetRunner:
    """
    Alpha Factory Testnet 运行时。

    行情数据来自实盘 WebSocket（数据真实）
    下单接口连接 Testnet（资金安全）
    """

    def __init__(self):
        if not TESTNET_API_KEY or not TESTNET_API_SECRET:
            raise ValueError(
                "请先设置 Testnet API Key：\n"
                "  set TESTNET_API_KEY=your_key\n"
                "  set TESTNET_API_SECRET=your_secret\n"
                "申请地址: https://testnet.binancefuture.com"
            )

        self.exec_engine   = TestnetEngine(TESTNET_API_KEY, TESTNET_API_SECRET)
        self.db            = Database("alpha_factory_trades.db")
        self.strategy      = None
        self.ws_feed       = None
        self.rest_fetcher  = None
        self._running      = False
        self._rank_thread  = None
        logger.info("[DB] 交易记录数据库已初始化 → alpha_factory_trades.db")

    def _on_agg_trade(self, data: dict):
        """
        WS 回调：直接更新特征引擎，绕过事件队列。

        原路径：WS线程 → Queue → EventEngine线程 → on_tick() → feature_engine
        新路径：WS线程 → feature_engine（消除线程切换和队列积压）
        """
        if self.strategy is None:
            return
        sym   = data["symbol"]
        price = data["price"]
        self.strategy.feature_engine.on_trade(
            symbol         = sym,
            price          = price,
            qty            = data["qty"],
            is_buyer_maker = data["is_buyer_maker"],
            ts_ms          = data["timestamp"],
        )
        self.strategy._tick_count += 1
        # 插针检测（成交维度）
        self.strategy.shock_detector.on_trade(
            symbol   = sym,
            price    = price,
            usdt_vol = price * data["qty"],
            ts_ms    = data["timestamp"],
        )

    def _on_book_ticker(self, data: dict):
        """WS 回调：直接更新特征引擎，绕过事件队列。"""
        if self.strategy is None:
            return
        self.strategy.feature_engine.on_book_ticker(
            symbol  = data["symbol"],
            bid     = data["bid"],
            bid_qty = data["bid_qty"],
            ask     = data["ask"],
            ask_qty = data["ask_qty"],
        )

    def _on_depth(self, data: dict):
        """WS 回调：@depth5@100ms → LOB 流形引擎（直接调用，绕过事件队列）。"""
        if self.strategy is None:
            return
        self.strategy.on_lob_depth(data)

    def _rank_loop(self):
        """
        独立排序线程：每秒检查一次，到达 rank_interval 时触发打分和交易决策。

        与 WS 回调线程分离：特征更新（高频，WS线程）和交易决策（低频，本线程）完全解耦。
        """
        interval = self.strategy.params["rank_interval"]
        while self._running:
            time.sleep(1)
            if not self._running:
                break
            now = time.time()
            if now - self.strategy._last_rank_time >= interval:
                try:
                    self.strategy._rank_and_trade(now)
                except Exception as e:
                    logger.error(f"[Runner] _rank_and_trade 异常: {e}")

    def _on_derivatives_update(self, symbol: str, funding_rate: float, oi: float):
        if self.strategy:
            self.strategy.update_derivatives(symbol, funding_rate, oi)

    async def start(self):
        logger.info("=" * 60)
        logger.info("  Alpha Factory - Binance Testnet 测试")
        logger.info("  行情: 实盘 WebSocket | 下单: Testnet API")
        logger.info("=" * 60)

        # 获取交易池（用实盘数据）
        all_symbols = await UniverseFilter(
            min_volume_usdt = 5_000_000,
            max_symbols     = 600,
        ).fetch_universe()

        # 过滤掉 Testnet 不存在的合约（避免 -1121 Invalid symbol）
        symbols = [s for s in all_symbols if s in self.exec_engine.valid_symbols]
        logger.info(
            f"交易池: 实盘 {len(all_symbols)} 个 → Testnet 有效 {len(symbols)} 个"
        )
        logger.info(f"前10: {symbols[:10]}")

        # 初始化策略（使用极小仓位）
        self.strategy = AlphaFactoryStrategy(
            engine  = self.exec_engine,
            symbols = symbols,
            db      = self.db,
            params  = {
                "rank_interval":        60,    # 1分钟排序一次
                "max_long_positions":    5,    # 最多做多5个
                "max_short_positions":   5,    # 最多做空5个
                "long_score_threshold":  0.5,
                "short_score_threshold":-0.5,
                "ema_alpha":             0.4,
                "confirm_rounds":        2,
                "leverage":             10,    # 与交易所杠杆设置一致
                "sl_vol_mult":         2.0,   # SL = vol×2.0×10x ≈ 4% 保证金
                "trade_size_usdt":      15.0,  # 15 USDT（精度截断后仍 > 5 USDT 门槛）
                "min_volume_zscore":     0.5,  # 降低门槛，testnet 只有20个币
                "warmup_count":          2,
                "max_spread_bps":       50.0,  # 放宽价差限制
            },
        )

        # 行情连接实盘（Testnet 行情太冷清）
        self._running = True
        self.ws_feed = MultiSymbolFeed(
            symbols        = symbols,
            on_agg_trade   = self._on_agg_trade,
            on_book_ticker = self._on_book_ticker,
            on_depth       = self._on_depth,
        )
        self.ws_feed.start()

        # 独立排序线程：特征更新（WS线程）与交易决策（本线程）分离
        self._rank_thread = threading.Thread(
            target=self._rank_loop, daemon=True, name="RankLoop"
        )
        self._rank_thread.start()

        self.rest_fetcher = RestFetcher(symbols, self._on_derivatives_update, interval=60)
        self.rest_fetcher.start()
        self.rest_fetcher.fetch_once()  # 立即预热一次

        self.strategy.on_start()
        logger.info("Testnet 测试运行中... Ctrl+C 停止")

        try:
            while True:
                await asyncio.sleep(60)
                # 每分钟同步查询 Testnet 真实持仓，验证下单是否生效
                real_positions = self.exec_engine.get_positions()
                if real_positions:
                    for p in real_positions:
                        logger.info(
                            f"[Testnet持仓] {p['symbol']}: "
                            f"qty={p['positionAmt']} "
                            f"pnl={p.get('unRealizedProfit', p.get('unrealizedProfit', '?'))}"
                        )
                else:
                    logger.info("[Testnet持仓] 暂无持仓")
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.stop()

    def stop(self):
        self._running = False
        self.strategy.on_stop()
        self.ws_feed.stop()
        self.rest_fetcher.stop()
        self.db.close()
        logger.info("Testnet 测试已停止")


if __name__ == "__main__":
    try:
        asyncio.run(AlphaFactoryTestnetRunner().start())
    except KeyboardInterrupt:
        pass

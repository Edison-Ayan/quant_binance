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
import argparse
import sys
import time
import threading

# Event 包装（供直接调用 strategy.on_tick / on_order_book 使用）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.event import Event
from core.constants import EventType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

from data_layer.logger import logger
from data_layer.multi_symbol_ws import MultiSymbolFeed
from data_layer.rest_fetcher import RestFetcher
from alpha_factory.universe_filter import UniverseFilter
from alpha_factory.alpha_strategy import AlphaFactoryStrategy
from alpha_factory.market_state_engine import MarketState, MarketRegime
from storage.database import Database
from monitor.report_engine import ReportEngine


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

    def _safe_stop_price(self, symbol: str, stop_price: float, side: str) -> float:
        """
        确保止损价与当前价有足够距离（Binance 要求 ≥ 0.1%，此处用 1.0% 保留余量）。
        side="SELL"（多头止损）：stop_price 必须低于当前价
        side="BUY" （空头止损）：stop_price 必须高于当前价

        当前价来源优先级：
          1. strategy._latest_features（最新 tick 价）
          2. stop_price 本身反推（兜底：强制在 stop_price 基础上再移动 MIN_DIST）
        """
        MIN_DIST = 0.010  # 1.0%，Binance testnet 要求比主网更严
        current = 0.0
        if hasattr(self, "strategy"):
            feat = self.strategy._latest_features.get(symbol)
            if feat and feat.last_price > 0:
                current = feat.last_price

        if current <= 0:
            # 无法获取当前价，从 stop_price 反推：直接在 stop_price 基础上再移 MIN_DIST
            if side == "SELL":
                adjusted = stop_price * (1 - MIN_DIST)
            else:
                adjusted = stop_price * (1 + MIN_DIST)
            logger.debug(f"[Testnet] {symbol} SL兜底调整 {stop_price:.6f}→{adjusted:.6f}（无当前价）")
            return adjusted

        if side == "SELL":
            max_allowed = current * (1 - MIN_DIST)
            if stop_price > max_allowed:
                logger.info(f"[Testnet] {symbol} SL调整 {stop_price:.6f}→{max_allowed:.6f} (距离不足，current={current:.6f})")
                return max_allowed
        else:
            min_allowed = current * (1 + MIN_DIST)
            if stop_price < min_allowed:
                logger.info(f"[Testnet] {symbol} SL调整 {stop_price:.6f}→{min_allowed:.6f} (距离不足，current={current:.6f})")
                return min_allowed
        return stop_price

    def _quantize_qty(self, symbol: str, qty: float) -> str:
        """按交易所 stepSize 截断数量，避免精度拒单"""
        decimals = self._step_sizes.get(symbol, 3)
        factor = 10 ** decimals
        truncated = int(qty * factor) / factor   # 向下截断
        return f"{truncated:.{decimals}f}"

    def _set_leverage(self, leverage: int = 10):
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
                # 确保止损价距当前价足够远，避免 Binance 以距离不足拒单
                stop_price = self._safe_stop_price(symbol, stop_price, side)
                formatted_price = self._format_price(symbol, stop_price)
                params = {
                    "symbol":        symbol,
                    "side":          side,
                    "type":          order_type,
                    "stopPrice":     formatted_price,
                    "closePosition": "true",
                    # CONTRACT_PRICE：用最新成交价验证/触发，避免 Testnet 上 MarkPrice 与
                    # LastPrice 偏差导致静默拒单（默认 MARK_PRICE 在 Testnet 上极不稳定）
                    "workingType":   "CONTRACT_PRICE",
                }
                try:
                    result = self.client.futures_create_order(**params)
                except BinanceAPIException as _sl_e:
                    logger.warning(
                        f"[Testnet] {symbol} {order_type} 挂单异常 "
                        f"stopPrice={formatted_price} code={_sl_e.code} msg={_sl_e.message}"
                    )
                    return None
                except Exception as _sl_e:
                    logger.warning(
                        f"[Testnet] {symbol} {order_type} 挂单未知异常 "
                        f"stopPrice={formatted_price} err={_sl_e}"
                    )
                    return None
                # Binance 有两种响应格式：
                # 1. 普通条件单 → {"orderId": 123, ...}
                # 2. Algo 条件单 → {"algoId": 123, "algoStatus": "NEW", ...}（同样成功）
                order_id  = result.get("orderId")
                algo_id   = result.get("algoId")
                effective_id = order_id or algo_id
                if effective_id is None:
                    logger.warning(
                        f"[Testnet] {symbol} {order_type} 挂单失败 "
                        f"stopPrice={formatted_price} full_result={result}"
                    )
                else:
                    id_type = "orderId" if order_id else "algoId"
                    logger.info(
                        f"[Testnet] {order_type} {side} {symbol} "
                        f"stopPrice={formatted_price} workingType=CONTRACT_PRICE "
                        f"→ {id_type}={effective_id}"
                    )
                    # 统一写回 orderId，供策略层的 _tp_sl_orders 追踪撤单使用
                    if order_id is None:
                        result["orderId"] = algo_id
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
                # 立即从策略层移除，防止每次 tick 重复触发
                if hasattr(self, "strategy"):
                    self.strategy._symbol_blacklist.add(symbol)
                    self.strategy.ranking_engine.release_long(symbol)
                    self.strategy.ranking_engine.release_short(symbol)
                    self.strategy.feature_engine.active_symbols.discard(symbol)
            else:
                logger.error(f"[Testnet] 下单失败 {symbol} {side} {qty} stop={stop_price}: {e}")
            return None

    def cancel_order(self, symbol: str, order_id: int):
        """
        撤销订单。Binance Testnet 上 STOP_MARKET closePosition 单被路由为 Algo 单（algoId），
        需要走 DELETE /fapi/v1/algo/futures/order 接口，与普通订单撤单接口不同。
        策略层保存的 orderId 可能实际是 algoId（send_order 中已统一写入 result["orderId"]），
        此处先尝试普通撤单，失败后自动尝试 algo 撤单。
        """
        try:
            self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"[Testnet] 撤单 {symbol} #{order_id}")
        except BinanceAPIException as e:
            if e.code == -2011:  # Unknown order — 可能是 algo 单
                try:
                    self.client.cancel_algo_order(symbol=symbol, algoId=order_id)
                    logger.info(f"[Testnet] algo撤单 {symbol} algoId={order_id}")
                except Exception as e2:
                    logger.debug(f"[Testnet] 撤单失败（可能已触发或不存在）{symbol} #{order_id}: {e2}")
            else:
                logger.debug(f"[Testnet] 撤单失败 {symbol} #{order_id}: {e}")

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

    def __init__(self, hold_forever: bool = False):
        self.hold_forever = hold_forever
        if not TESTNET_API_KEY or not TESTNET_API_SECRET:
            raise ValueError(
                "请先设置 Testnet API Key：\n"
                "  set TESTNET_API_KEY=your_key\n"
                "  set TESTNET_API_SECRET=your_secret\n"
                "申请地址: https://testnet.binancefuture.com"
            )

        self.exec_engine    = TestnetEngine(TESTNET_API_KEY, TESTNET_API_SECRET)
        self.db             = Database("alpha_factory_trades.db")
        self.report_engine  = ReportEngine()
        self.strategy       = None
        self.ws_feed        = None
        self.rest_fetcher   = None
        self._running       = False
        self._rank_thread   = None
        logger.info("[DB] 交易记录数据库已初始化 → alpha_factory_trades.db")

    def _on_agg_trade(self, data: dict):
        """
        WS 回调 → strategy.on_tick()（含完整快层决策逻辑）。

        直接构造 Event 并调用 on_tick，确保：
        - 特征更新、插针检测、trailing stop、take_profit、候选池检查、开仓触发
        全部在同一调用链中完成，与 Paper Trading 模式行为一致。

        _external_rank_loop=True 保证 on_tick 内部不会重复触发 _rank_and_trade
        （排序由独立 _rank_loop 线程负责）。
        """
        if self.strategy is None:
            return
        self.strategy.on_tick(Event(EventType.TICK, {
            "symbol":         data["symbol"],
            "price":          data["price"],
            "qty":            data["qty"],
            "is_buyer_maker": data["is_buyer_maker"],
            "timestamp":      data["timestamp"],
        }))

    def _on_book_ticker(self, data: dict):
        """
        WS 回调 → strategy.on_order_book()（含 timing_engine 和 shock_detector 更新）。

        之前直接调用 feature_engine.on_book_ticker() 会漏掉：
        - timing_engine.on_book_ticker() → microprice_delta 无法计算 → timing_score ≈ 0
        - shock_detector.on_book()       → 盘口冲击检测失效
        """
        if self.strategy is None:
            return
        bid = data["bid"]
        ask = data["ask"]
        self.strategy.on_order_book(Event(EventType.ORDER_BOOK, {
            "symbol":       data["symbol"],
            "best_bid":     bid,
            "best_bid_qty": data["bid_qty"],
            "best_ask":     ask,
            "best_ask_qty": data["ask_qty"],
            "bids":         [[bid, data["bid_qty"]]],
            "asks":         [[ask, data["ask_qty"]]],
            "timestamp":    int(time.time() * 1000),
        }))

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

    def _on_derivatives_update(self, symbol: str, funding_rate: float, oi: float, ret_24h: float = 0.0):
        if self.strategy:
            self.strategy.update_derivatives(symbol, funding_rate, oi, ret_24h)

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
                "stop_loss_pct":       0.008,  # 交易所SL挂在0.8%价格 = 8%保证金（10x）
                "sl_vol_mult":         6.0,   # vol×6，被stop_loss_pct覆盖
                "max_single_loss_margin_pct": 0.08,  # 保证金亏8%（=价格0.8%）程序端强制平仓
                "take_profit_pct":     0.015,  # 固定止盈：价格1.5% = 保证金15%（10x）
                "lifecycle_exit":      True,   # 启用lifecycle驱动退出（REVERSAL平仓）
                "alpha_trailing_min_pnl": 999.0,  # 暂停alpha trailing（阈值设极大使其不触发）
                "timing_exit_threshold": 999.0,  # 暂停alpha flip（阈值设极大使其不触发）
                "evolve_interval":    99999,   # 暂停因子权重自动进化
                "trailing_vol_mult":    0.0,    # 暂停trailing stop（设为0跳过此逻辑）
                "trade_size_usdt":      15.0,  # 15 USDT（精度截断后仍 > 5 USDT 门槛）
                "min_volume_zscore":     0.3,  # 关闭成交量门槛（testnet 行情量极低）
                "warmup_count":          2,
                "max_spread_bps":       50.0,  # 放宽价差限制
                "kill_switch_enabled":     False, # 测试阶段关闭 Kill Switch
                "shock_detector_enabled": False, # 测试阶段关闭 ShockDetector
                "hold_forever":         self.hold_forever,  # 不平仓模式
            },
        )

        # ── 暂停 LOB PCA：lob_engine 替换为哑对象，timing_engine LOB权重清零 ──────
        class _NoopLOBEngine:
            def on_order_book(self, *a, **kw): pass
            def get_symbol_latent(self, sym): return None
            def get_symbol_bucket(self, sym): return "mid"
            def get_symbol_pc1(self, sym): return 0.0
        self.strategy.lob_engine = _NoopLOBEngine()
        # timing_score 只保留 microprice + OFI，LOB PCA 分量权重全部清零
        te = self.strategy.timing_engine
        te.w_lob_pc1    = 0.0
        te.w_lob_z1     = 0.0
        te.w_lob_z2     = 0.0
        te.w_lob_z3     = 0.0
        # 重新归一化：microprice(0.30) + ofi(0.25) → 各自等比放大至合计1.0
        te.w_microprice = 0.55
        te.w_ofi        = 0.45

        # ── 暂停 MarketStateEngine：替换为永远放行的哑对象 ──────────────────────
        _always_tradeable = MarketState(
            regime       = MarketRegime.TRENDING,
            dispersion   = 1.0,
            tradability  = 1.0,
            crowding_score = 0.0,
            is_tradeable = True,
            regime_mult  = 1.0,
            long_bias    = 0.0,
        )
        class _PassthroughMSE:
            def update(self, features):
                return _always_tradeable
            def get_state(self):
                return _always_tradeable
        self.strategy.market_state_engine = _PassthroughMSE()

        # rank_loop 线程独立负责排序，on_tick 内部不再触发 _rank_and_trade
        self.strategy._external_rank_loop = True

        # 行情连接实盘（Testnet 行情太冷清）
        self._running = True
        self.ws_feed = MultiSymbolFeed(
            symbols        = symbols,
            on_agg_trade   = self._on_agg_trade,
            on_book_ticker = self._on_book_ticker,
            on_depth       = self._on_depth,
        )
        self.ws_feed.start()
        # 将 depth5 动态订阅函数注入策略（每轮排序后自动更新 TopN + 持仓的 depth）
        self.strategy._depth_update_fn = self.ws_feed.update_depth_symbols

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

                # 向 ReportEngine 记录市场状态快照
                ms = getattr(self.strategy, "_latest_market_state", None)
                if ms:
                    self.report_engine.record_market_state(
                        regime      = ms.regime.value,
                        tradability = ms.tradability,
                        dispersion  = ms.dispersion,
                    )
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.stop()

    def stop(self):
        self._running = False
        if self.strategy:
            self.strategy.on_stop()
        if self.ws_feed:
            self.ws_feed.stop()
        if self.rest_fetcher:
            self.rest_fetcher.stop()

        # 生成 session 级别运行报告（在 db.close() 之前，确保队列已刷完）
        self.report_engine.generate(
            strategy = self.strategy,
            logger   = logger,
        )

        self.db.close()
        logger.info("Testnet 测试已停止")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Factory Testnet")
    parser.add_argument(
        "--hold-forever",
        action="store_true",
        default=False,
        help="买入后不平仓（禁用所有止损止盈，调试用）",
    )
    args = parser.parse_args()
    try:
        asyncio.run(AlphaFactoryTestnetRunner(hold_forever=args.hold_forever).start())
    except KeyboardInterrupt:
        pass

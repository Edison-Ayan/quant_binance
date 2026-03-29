"""
Alpha Factory Live Runner - 全市场异动捕捉系统入口

运行模式：
    Paper Trading (默认) : 不连接真实交易所，只打印信号，验证系统逻辑
    Live Trading         : 连接 Binance Futures，真实下单（需配置 API Key）

启动步骤：
    1. 从 Binance 获取全市场交易池（筛选 USDT 永续合约）
    2. 初始化事件引擎 + AlphaFactory 策略
    3. 启动 MultiSymbolFeed（aggTrade + bookTicker WebSocket）
    4. 启动 RestFetcher（资金费率 + OI 轮询）
    5. 进入主循环（定期打印状态）

使用方式：
    # Paper Trading（推荐先验证）
    python live_trading/run_alpha_factory.py

    # Live Trading
    BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy python live_trading/run_alpha_factory.py --live

架构图：
    MultiSymbolFeed(WS)  →  EventEngine  →  AlphaFactoryStrategy
    RestFetcher(REST)    ↗                      ↓
                                          FeatureEngine
                                                ↓
                                          ScoringEngine
                                                ↓
                                          RankingEngine
                                                ↓
                                     MockEngine / BinanceGateway
"""

import asyncio
import argparse
import signal
import sys
import time

# 将项目根目录加入 Python 路径
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config
from core.event import Event
from core.constants import EventType
from core.event_engine import EventEngine
from data_layer.logger import logger
from data_layer.multi_symbol_ws import MultiSymbolFeed
from data_layer.rest_fetcher import RestFetcher

from alpha_factory.universe_filter import UniverseFilter
from alpha_factory.alpha_strategy import AlphaFactoryStrategy


# ── Paper Trading 模拟引擎 ────────────────────────────────────────────────────

class PaperEngine:
    """
    Paper Trading 模拟引擎。

    替代真实的 StrategyEngine，记录所有模拟成交，
    方便在不承担真实风险的情况下验证策略逻辑。
    """

    def __init__(self):
        self.trades     = []
        self.total_pnl  = 0.0

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
        order_id = len(self.trades) + 1
        _STOP_TYPES = ("STOP_MARKET", "TAKE_PROFIT_MARKET")
        if order_type in _STOP_TYPES:
            tag = "TP" if order_type == "TAKE_PROFIT_MARKET" else "SL"
            logger.info(
                f"[PAPER] {tag} {side} {symbol} stopPrice={stop_price}  (order #{order_id})"
            )
        else:
            self.trades.append({
                "id":          order_id,
                "symbol":      symbol,
                "side":        side,
                "qty":         qty,
                "order_type":  order_type,
                "reduce_only": reduce_only,
                "timestamp":   time.time(),
            })
            tag = "CLOSE" if reduce_only else "OPEN"
            logger.info(
                f"[PAPER] {tag} {side} {qty:.6f} {symbol}  (order #{order_id})"
            )
        return {"orderId": order_id}

    def cancel_order(self, symbol: str, order_id: int):
        logger.info(f"[PAPER] CANCEL #{order_id} {symbol}")

    def get_summary(self) -> dict:
        return {
            "total_trades": len(self.trades),
            "buys":         sum(1 for t in self.trades if t["side"] == "BUY"),
            "sells":        sum(1 for t in self.trades if t["side"] == "SELL"),
        }


# ── 主引擎 ────────────────────────────────────────────────────────────────────

class AlphaFactoryRunner:
    """
    Alpha Factory 完整运行时。

    负责整合所有组件的生命周期管理：
    启动 → 运行 → 优雅停止
    """

    def __init__(self, live: bool = False):
        self.live         = live
        self.event_engine = EventEngine()
        self.strategy:    AlphaFactoryStrategy = None
        self.ws_feed:     MultiSymbolFeed       = None
        self.rest_fetcher: RestFetcher          = None

    # ─── WebSocket 回调（数据层 → 事件层）──────────────────────────────────────

    def _on_agg_trade(self, data: dict):
        """MultiSymbolFeed 的 aggTrade 回调 → 包装为 TICK 事件投递"""
        event = Event(EventType.TICK, {
            "symbol":          data["symbol"],
            "price":           data["price"],
            "qty":             data["qty"],
            "is_buyer_maker":  data["is_buyer_maker"],
            "timestamp":       data["timestamp"],
        })
        self.event_engine.put(event)

    def _on_book_ticker(self, data: dict):
        """MultiSymbolFeed 的 bookTicker 回调 → 包装为 ORDER_BOOK 事件投递"""
        event = Event(EventType.ORDER_BOOK, {
            "symbol":       data["symbol"],
            "best_bid":     data["bid"],
            "best_bid_qty": data["bid_qty"],
            "best_ask":     data["ask"],
            "best_ask_qty": data["ask_qty"],
            "bids":         [[data["bid"], data["bid_qty"]]],
            "asks":         [[data["ask"], data["ask_qty"]]],
            "timestamp":    int(time.time() * 1000),
        })
        self.event_engine.put(event)

    def _on_depth(self, data: dict):
        """MultiSymbolFeed 的 depth5 回调 → 直接喂给策略 LOB 引擎（绕过事件引擎减延迟）"""
        if self.strategy:
            self.strategy.on_lob_depth(data)

    def _on_derivatives_update(self, symbol: str, funding_rate: float, oi: float):
        """RestFetcher 的更新回调 → 直接调用策略（不经过事件引擎，减少延迟）"""
        if self.strategy:
            self.strategy.update_derivatives(symbol, funding_rate, oi)

    # ─── 主启动流程 ──────────────────────────────────────────────────────────

    async def start(self):
        mode_str = "🔴 LIVE TRADING" if self.live else "📄 PAPER TRADING"
        logger.info(f"{'='*60}")
        logger.info(f"  Alpha Factory 全市场异动捕捉系统 - {mode_str}")
        logger.info(f"{'='*60}")

        # ── 1. 获取交易池 ────────────────────────────────────────────────────
        logger.info("⏳ 正在获取全市场交易池...")
        universe_filter = UniverseFilter(
            min_volume_usdt = 5_000_000,   # 提高门槛可缩小池子，降低可扩大
            max_symbols     = 500,
        )
        symbols = await universe_filter.fetch_universe()
        logger.info(f"✅ 交易池: {len(symbols)} 个品种")
        logger.info(f"   前10名: {', '.join(symbols[:10])}")

        # ── 2. 初始化执行引擎 ─────────────────────────────────────────────────
        if self.live:
            if not config.validate():
                logger.error("❌ API Key 未配置，无法启动实盘！使用 --paper 模式或配置环境变量。")
                return
            logger.error("实盘模式尚未接入，请使用 Paper Trading 验证逻辑")
            return
        else:
            execution_engine = PaperEngine()

        # ── 3. 初始化策略 ─────────────────────────────────────────────────────
        self.strategy = AlphaFactoryStrategy(
            engine  = execution_engine,
            symbols = symbols,
            params  = {
                "rank_interval":        60,    # 每60秒重排序
                "max_long_positions":    3,    # 最多同时做多3个
                "max_short_positions":   3,    # 最多同时做空3个
                "long_score_threshold":  0.5,  # 做多最低 EMA 分数
                "short_score_threshold":-0.5,  # 做空最高 EMA 分数（负值）
                "ema_alpha":             0.4,  # EMA 平滑系数
                "confirm_rounds":        2,    # 连续上榜2次才进场
                "leverage":             10,    # 杠杆倍数
                "stop_loss_pct":        0.05,  # 保证金亏损5%止损
                "take_profit_pct":      0.10,  # 保证金盈利10%止盈
                "trade_size_usdt":     100.0,  # 每笔100 USDT
                "min_volume_zscore":    0.8,   # 成交量 Z-score 门槛
                "warmup_count":          4,    # 热身期4次排序
                "max_spread_bps":       20.0,  # 最大价差20bps
            },
        )

        # ── 4. 注册事件处理器 ──────────────────────────────────────────────────
        self.event_engine.register(EventType.TICK,       self.strategy.on_tick)
        self.event_engine.register(EventType.ORDER_BOOK, self.strategy.on_order_book)

        # ── 5. 启动事件引擎 ────────────────────────────────────────────────────
        self.event_engine.start()

        # ── 6. 启动多币 WebSocket ──────────────────────────────────────────────
        self.ws_feed = MultiSymbolFeed(
            symbols        = symbols,
            on_agg_trade   = self._on_agg_trade,
            on_book_ticker = self._on_book_ticker,
            on_depth       = self._on_depth,
        )
        self.ws_feed.start()
        logger.info("✅ WebSocket 数据流已启动")

        # ── 7. 启动 REST 轮询（资金费率 + OI）──────────────────────────────────
        self.rest_fetcher = RestFetcher(
            symbols   = symbols,
            on_update = self._on_derivatives_update,
            interval  = 60,
        )
        self.rest_fetcher.start()
        logger.info("✅ REST 数据轮询已启动")

        # ── 8. 启动策略 ────────────────────────────────────────────────────────
        self.strategy.on_start()
        logger.info(f"\n{'='*60}")
        logger.info("  系统启动完成，开始监控全市场...")
        logger.info(f"  每 30s 打印一次状态 | Ctrl+C 优雅停止")
        logger.info(f"{'='*60}\n")

        # ── 9. 主循环 ──────────────────────────────────────────────────────────
        try:
            while True:
                await asyncio.sleep(30)
                self._print_status(execution_engine)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("\n⏹ 收到停止信号，开始优雅关闭...")
            self.stop(execution_engine)

    def _print_status(self, execution_engine: PaperEngine):
        """每30秒打印一次系统状态"""
        if not self.strategy:
            return

        status = self.strategy.get_status()
        summary = execution_engine.get_summary()

        longs  = list(status["long_positions"].keys())  or ["无"]
        shorts = list(status["short_positions"].keys()) or ["无"]
        logger.info(
            f"\n── 系统状态 ──────────────────────────────────────\n"
            f"  监控品种:  {status['active_symbols']}\n"
            f"  排序次数:  {status['rank_count']}\n"
            f"  多头持仓:  {longs}\n"
            f"  空头持仓:  {shorts}\n"
            f"  多头Top N: {status['long_top']}\n"
            f"  空头Top N: {status['short_top']}\n"
            f"  成交笔数:  买={summary['buys']} 卖={summary['sells']}\n"
            f"──────────────────────────────────────────────────"
        )

    def stop(self, execution_engine: PaperEngine = None):
        """优雅停止所有组件"""
        if self.strategy:
            self.strategy.on_stop()
        if self.ws_feed:
            self.ws_feed.stop()
        if self.rest_fetcher:
            self.rest_fetcher.stop()
        self.event_engine.stop()

        if execution_engine:
            summary = execution_engine.get_summary()
            logger.info(
                f"\n{'='*60}\n"
                f"  Alpha Factory 已停止\n"
                f"  总成交笔数: {summary['total_trades']}\n"
                f"  买入: {summary['buys']} | 卖出: {summary['sells']}\n"
                f"{'='*60}"
            )


# ── 程序入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alpha Factory - 全市场异动捕捉系统")
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="启用实盘模式（默认为 Paper Trading）",
    )
    args = parser.parse_args()

    runner = AlphaFactoryRunner(live=args.live)

    # 注册 Ctrl+C 信号处理
    def handle_signal(signum, frame):
        logger.info("收到中断信号...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)

    asyncio.run(runner.start())


if __name__ == "__main__":
    main()

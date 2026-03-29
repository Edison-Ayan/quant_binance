"""
REST Data Fetcher - 资金费率和持仓量轮询

职责：
    通过 Binance Futures REST API 定期拉取无法从 WebSocket 获取的数据：
    - 资金费率（lastFundingRate）
    - 持仓量（openInterest）

    结果通过回调函数传递给 AlphaFactoryStrategy.update_derivatives()。

数据更新频率：
    资金费率: 每 60 秒批量拉取（1次请求获取全市场）
    持仓量:   每 60 秒分批拉取（每品种1次，并发执行）

使用方式：
    def on_update(symbol, funding_rate, oi):
        strategy.update_derivatives(symbol, funding_rate, oi)

    fetcher = RestFetcher(symbols, on_update, interval=60)
    fetcher.start()

注意事项：
    - OI 数据每次需要为每个品种发一个独立请求
    - Binance 有请求频率限制（默认 1200 权重/分钟）
    - 并发数量默认 10，避免触发限速
"""

import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from typing import Callable, List, Optional

from data_layer.logger import logger


FAPI_BASE   = "https://fapi.binance.com"
MAX_WORKERS = 20    # 并发请求数（OI 每个品种独立请求，提高并发缩短轮询时间）
TIMEOUT     = 5     # 单次请求超时（秒）
# 连接池大小：与 MAX_WORKERS 对齐，复用 TCP+SSL 连接，消除每次握手开销
_POOL_SIZE  = MAX_WORKERS + 4


class RestFetcher:
    """
    Binance Futures REST 数据轮询器。

    在独立线程中定期执行：
    1. 批量拉取资金费率（1次请求，高效）
    2. 并发拉取各品种持仓量（每品种1次，使用线程池）
    3. 合并后通过 on_update 回调传递给策略层
    """

    def __init__(
        self,
        symbols:    List[str],
        on_update:  Callable[[str, float, float], None],   # (symbol, funding_rate, oi)
        interval:   int = 60,
    ):
        """
        参数：
            symbols   : 需要轮询的交易对列表
            on_update : 数据更新回调，签名：(symbol: str, funding_rate: float, oi: float)
            interval  : 轮询间隔（秒），默认 60s
        """
        self.symbols   = symbols
        self.on_update = on_update
        self.interval  = interval
        self._running  = False
        self._thread: Optional[threading.Thread] = None

        # 复用连接池（避免每次 OI 请求重新 TCP+SSL 握手）
        adapter = HTTPAdapter(
            pool_connections = _POOL_SIZE,
            pool_maxsize     = _POOL_SIZE,
            max_retries      = 1,
        )
        self._session = requests.Session()
        self._session.mount("https://", adapter)

    # ─── 公开接口 ────────────────────────────────────────────────────────────

    def start(self):
        """启动后台轮询线程"""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="RestFetcher",
        )
        self._thread.start()
        logger.info(
            f"[RestFetcher] 启动 | 品种数={len(self.symbols)} | 间隔={self.interval}s"
        )

    def stop(self):
        """停止轮询"""
        self._running = False
        logger.info("[RestFetcher] 已停止")

    def fetch_once(self):
        """立即执行一次全量拉取（同步，可在启动时预热调用）"""
        self._fetch_and_dispatch()

    # ─── 内部实现 ────────────────────────────────────────────────────────────

    def _run_loop(self):
        """轮询循环（在后台线程中运行）"""
        # 启动时立即执行一次，避免等待第一个 interval
        self._fetch_and_dispatch()
        while self._running:
            time.sleep(self.interval)
            if self._running:
                self._fetch_and_dispatch()

    def _fetch_and_dispatch(self):
        """执行一次完整的数据拉取并触发回调"""
        try:
            # Step 1: 批量拉取资金费率（1次请求获取全市场）
            funding_map = self._fetch_funding_rates()

            # Step 2: 并发拉取各品种持仓量
            oi_map = self._fetch_open_interests()

            # Step 3: 合并触发回调
            dispatched = 0
            for sym in self.symbols:
                fr = funding_map.get(sym, 0.0)
                oi = oi_map.get(sym, 0.0)
                try:
                    self.on_update(sym, fr, oi)
                    dispatched += 1
                except Exception as e:
                    logger.error(f"[RestFetcher] on_update 回调错误 {sym}: {e}")

            logger.debug(f"[RestFetcher] 更新完成 | {dispatched} 个品种")

        except Exception as e:
            logger.error(f"[RestFetcher] 轮询失败: {e}")

    def _fetch_funding_rates(self) -> dict:
        """
        批量拉取资金费率。

        接口：GET /fapi/v1/premiumIndex
        返回：{symbol: last_funding_rate}

        1次请求即可获取全市场所有品种的资金费率，效率高。
        """
        try:
            resp = self._session.get(
                f"{FAPI_BASE}/fapi/v1/premiumIndex",
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                item["symbol"]: float(item.get("lastFundingRate", 0) or 0)
                for item in data
                if "symbol" in item
            }
        except Exception as e:
            logger.warning(f"[RestFetcher] 资金费率拉取失败: {e}")
            return {}

    def _fetch_open_interests(self) -> dict:
        """
        并发拉取各品种持仓量。

        接口：GET /fapi/v1/openInterest?symbol=BTCUSDT
        Binance 无批量持仓量接口，需逐个品种请求。
        使用 ThreadPoolExecutor 并发执行，控制并发数避免限速。
        """
        result = {}

        def fetch_one(symbol: str) -> tuple:
            try:
                resp = self._session.get(
                    f"{FAPI_BASE}/fapi/v1/openInterest",
                    params={"symbol": symbol},
                    timeout=TIMEOUT,
                )
                resp.raise_for_status()
                oi = float(resp.json().get("openInterest", 0) or 0)
                return symbol, oi
            except Exception:
                return symbol, 0.0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_one, sym): sym for sym in self.symbols}
            for future in as_completed(futures):
                sym, oi = future.result()
                result[sym] = oi

        return result

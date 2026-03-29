"""
Universe Filter - 全市场交易池筛选

筛选条件：
    1. 必须是 USDT 结算的永续合约
    2. 24h USDT 成交量 > min_volume_usdt
    3. 最多返回 max_symbols 个（按成交量降序）

使用方式：
    filter = UniverseFilter()
    symbols = await filter.fetch_universe()
    # → ["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]

网络说明：
    使用 requests 而非 aiohttp，因为 requests 自动使用系统代理，
    aiohttp 默认不走系统代理，在代理环境下会连接超时。
    fetch_universe 仍保持 async 接口（用 asyncio.get_event_loop().run_in_executor
    在线程池中执行同步请求），调用方无需修改。
"""

import asyncio
import requests
from typing import List, Tuple
from data_layer.logger import logger


class UniverseFilter:

    FAPI_BASE = "https://fapi.binance.com"

    def __init__(
        self,
        min_volume_usdt: float = 5_000_000,
        max_symbols:     int   = 500,
    ):
        self.min_volume_usdt = min_volume_usdt
        self.max_symbols     = max_symbols

    async def fetch_universe(self) -> List[str]:
        """异步接口，内部用线程池执行同步 requests（自动走系统代理）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_sync)

    def _fetch_sync(self) -> List[str]:
        """同步拉取，使用 requests（自动读取系统代理环境变量）"""
        resp = requests.get(
            f"{self.FAPI_BASE}/fapi/v1/ticker/24hr",
            timeout=10,
        )
        resp.raise_for_status()
        tickers = resp.json()

        # TradFi 合约前缀（需签署额外协议，普通账号无法交易）
        _TRADFI_PREFIXES = ("XAU", "XAG", "XPT", "XPD", "PAXG", "AMZN", "TSLA",
                            "AAPL", "GOOG", "MSFT", "NVDA", "META", "NFLX")

        candidates: List[Tuple[str, float]] = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            if any(sym.startswith(p) for p in _TRADFI_PREFIXES):
                continue
            volume = float(t.get("quoteVolume", 0))
            if volume < self.min_volume_usdt:
                continue
            candidates.append((sym, volume))

        candidates.sort(key=lambda x: -x[1])
        result = [sym for sym, _ in candidates[:self.max_symbols]]

        logger.info(
            f"[UniverseFilter] 筛选结果: {len(result)}/{len(tickers)} 个币种 "
            f"(成交量>{self.min_volume_usdt/1e6:.0f}M)"
        )
        return result

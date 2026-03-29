"""
统一数据加载与采集模块（DataLoader）

职责：
    整合所有数据下载、解析和加载功能，提供统一接口：

    ── K 线 / 行情 ──────────────────────────────────────────────────────────
    1. load_from_csv()           : 从本地 CSV 加载标准 OHLCV 数据
    2. download_binance_data()   : 从 Binance K 线 API 下载历史数据
    3. download_agg_trades()     : 从 Binance aggTrades API 下载逐笔成交
    4. resample_data()           : OHLCV 重采样（1m→15m 等）
    5. add_technical_indicators(): 添加 MA/RSI/MACD/布林带指标
    6. validate_data()           : 数据质量验证

    ── Binance Vision 历史 LOB ──────────────────────────────────────────────
    7. download_lob_history()    : 下载 Binance Vision bookDepth+trades zip
                                   并计算 20 维 LOB 特征，保存 .npy 文件
    8. parse_bookdepth_zip()     : 解析 bookDepth zip → 快照列表
    9. parse_trades_zip()        : 解析 trades zip → (M,3) numpy 数组
   10. compute_lob_features()    : 从快照+成交流水计算 20 维特征矩阵

    ── REST API 实时 LOB 采集 ───────────────────────────────────────────────
   11. fetch_depth()             : 单次拉取订单簿快照
   12. fetch_recent_trades()     : 单次拉取最近成交
   13. collect_rest_lob()        : 持续轮询采集并保存 .npy 文件
                                  （对应 RestLOBCollector.collect()）

    ── Hawkes 训练数据加载 ──────────────────────────────────────────────────
   14. load_trades_csv()         : 加载单个 trades CSV → Hawkes 事件列表
   15. load_all_trades()         : 加载多个 CSV (glob/列表) → 合并事件列表

    ── 特征矩阵加载 ─────────────────────────────────────────────────────────
   16. load_lob_features()       : 加载 .npy 特征矩阵 + 方向标签，打印统计

模块级工具函数（可独立调用，无需实例化 DataLoader）：
    url_exists()         : HEAD 请求检查 Binance Vision URL 是否存在
    probe_latest_date()  : 自动探测最新可用的 bookDepth 日期
    download_file()      : 通用 HTTP 文件下载（带进度，支持断点检测）
    parse_bookdepth_zip(): 同上 #8
    parse_trades_zip()   : 同上 #9
    compute_lob_features(): 同上 #10
    fetch_depth()        : 同上 #11
    fetch_recent_trades(): 同上 #12

模块级类：
    RestLOBCollector     : REST 轮询订单簿采集器（线程安全，带后台成交线程）
"""

import io
import time
import glob as _glob_module
import zipfile
import threading
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import os

from data_layer.logger import logger


# ══════════════════════════════════════════════════════════════════════════════
# LOB 特征名（与 lob_features.py 保持一致）
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "spread_bps", "micro_dev_bps",
    "imbalance_L1", "imbalance_L5", "imbalance_L10", "imbalance_weighted",
    "depth_bid_log", "depth_ask_log", "depth_ratio", "top_depth_ratio",
    "slope_bid_bps", "slope_ask_bps",
    "vol_decay_bid", "vol_decay_ask",
    "ofi_lob_100ms", "ofi_lob_500ms",
    "ofi_trade_100ms", "ofi_trade_500ms", "ofi_trade_1s",
    "trade_rate_1s",
]
N_LOB_FEATURES = len(FEATURE_NAMES)   # 20


# ══════════════════════════════════════════════════════════════════════════════
# Binance Vision URL 工具
# ══════════════════════════════════════════════════════════════════════════════

_BASE_FUTURES = "https://data.binance.vision/data/futures/um/daily"
_FAPI_BASE    = "https://fapi.binance.com"
_SPOT_BASE    = "https://api.binance.com"

_BID_BANDS = [-0.2, -1.0, -2.0, -3.0, -4.0, -5.0]   # 由近到远
_ASK_BANDS = [ 0.2,  1.0,  2.0,  3.0,  4.0,  5.0]


def _url_bookdepth(symbol: str, date_str: str) -> str:
    return f"{_BASE_FUTURES}/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"


def _url_trades(symbol: str, date_str: str) -> str:
    return f"{_BASE_FUTURES}/trades/{symbol}/{symbol}-trades-{date_str}.zip"


# ──────────────────────────────────────────────────────────────────────────────
# 模块级工具函数：下载
# ──────────────────────────────────────────────────────────────────────────────

def url_exists(url: str) -> bool:
    """
    HEAD 请求检查 URL 是否存在（404 返回 False，其他异常也返回 False）。

    用途：在下载前探测 Binance Vision 上某天的数据是否已发布。
    """
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False


def probe_latest_date(symbol: str, max_lookback: int = 7) -> Optional[str]:
    """
    从昨天往前最多 max_lookback 天，找到 Binance Vision 上最新可用的 bookDepth 日期。

    Binance Vision 通常有 1-2 天的发布延迟（今天的数据明天才能下到）。

    返回：'YYYY-MM-DD' 字符串，或 None（未找到）。
    """
    today = datetime.now(timezone.utc).date()
    for d in range(1, max_lookback + 1):
        date_str = (today - timedelta(days=d)).strftime("%Y-%m-%d")
        url = _url_bookdepth(symbol, date_str)
        logger.info(f"  探测 {date_str} ... {url}")
        if url_exists(url):
            logger.info(f"  最新可用日期: {date_str}")
            return date_str
    return None


def download_file(url: str, dest: str, show_progress: bool = True,
                  required: bool = True) -> bool:
    """
    下载 URL 内容到本地文件 dest。

    参数：
        url           : 下载地址
        dest          : 本地保存路径
        show_progress : 是否打印下载进度（默认 True）
        required      : True=失败时抛出 RuntimeError；False=静默返回 False

    若文件已存在则跳过下载（断点续传的简化实现）。
    """
    if os.path.exists(dest):
        logger.info(f"  已存在，跳过下载: {dest}")
        return True
    logger.info(f"  下载 {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total      = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk      = 1 << 20   # 1 MB per read
            buf        = io.BytesIO()
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                buf.write(data)
                downloaded += len(data)
                if show_progress and total:
                    pct = downloaded / total * 100
                    print(f"\r    {downloaded/1e6:.1f}/{total/1e6:.1f} MB  {pct:.0f}%",
                          end="", flush=True)
        if show_progress:
            print()
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as f:
            f.write(buf.getvalue())
        logger.info(f"  已保存 → {dest}")
        return True
    except Exception as e:
        msg = f"下载失败 {url}: {e}"
        if required:
            raise RuntimeError(msg)
        logger.warning(f"  {msg}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 模块级工具函数：解析 Binance Vision zip
# ──────────────────────────────────────────────────────────────────────────────

def parse_bookdepth_zip(path: str) -> list:
    """
    解析 Binance Vision bookDepth zip（累计深度图格式）。

    实际 CSV 格式（4列）：
        timestamp, percentage, depth, notional
        "2026-03-16 00:00:08", -5.00, 6766.48, 481030611.87
        ...
        每个时间戳 12 行：bid侧 -5/-4/-3/-2/-1/-0.2，ask侧 +0.2/+1/+2/+3/+4/+5
        depth/notional 均为从 mid 到该百分比的 **累计** 值。

    返回：
        snapshots : list of dict {
            'ts_ms' : int,
            'bands' : {pct_float: {'depth': float, 'notional': float}}
        }
    """
    from collections import defaultdict

    def _ts_to_ms(s: str) -> int:
        s = s.strip()
        if s.isdigit():
            t = int(s)
            return t if t > 1e12 else t * 1000
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

    raw_by_ts = defaultdict(dict)
    with zipfile.ZipFile(path) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            f.readline()   # header
            for line in f:
                parts = line.decode().strip().split(",")
                if len(parts) < 4:
                    continue
                try:
                    ts_ms    = _ts_to_ms(parts[0])
                    pct      = float(parts[1])
                    depth    = float(parts[2])
                    notional = float(parts[3])
                    raw_by_ts[ts_ms][pct] = {"depth": depth, "notional": notional}
                except (ValueError, IndexError):
                    continue

    snapshots = [{"ts_ms": ts, "bands": bands}
                 for ts, bands in sorted(raw_by_ts.items())]
    n = len(snapshots)
    avg_interval = (
        (snapshots[-1]["ts_ms"] - snapshots[0]["ts_ms"]) / max(n - 1, 1) / 1000
        if n > 1 else 0
    )
    logger.info(f"  bookDepth 解析完成: {n} 条快照  "
                f"平均间隔 {avg_interval:.1f}s  每天约 {86400/max(avg_interval,1):.0f} 条")
    return snapshots


def parse_trades_zip(path: str) -> np.ndarray:
    """
    解析 Binance Vision trades zip 文件。

    CSV 格式：
        id, price, qty, quoteQty, time, is_buyer_maker, is_best_match

    返回：
        trades : np.ndarray  shape (M, 3)  —  [time_ms, qty, sign]
            sign = +1 if not is_buyer_maker (主动买) else -1 (主动卖)
    """
    rows = []
    with zipfile.ZipFile(path) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            f.readline()   # header
            for raw in f:
                line = raw.decode().strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                ts   = int(parts[4])
                qty  = float(parts[2])
                ibm  = parts[5].strip().lower() in ("true", "1")
                sign = -1 if ibm else 1
                rows.append((ts, qty, sign))
    arr = np.array(rows, dtype=np.float64)
    logger.info(f"  trades 解析完成: {len(arr)} 条成交")
    return arr   # (M, 3)


# ──────────────────────────────────────────────────────────────────────────────
# LOB 特征计算内部辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _get_band(bands: dict, pct: float) -> Tuple[float, float]:
    b = bands.get(pct, None)
    if b is None:
        return 0.0, 0.0
    return b["depth"], b["notional"]


def _imbalance(bid_d: float, ask_d: float) -> float:
    total = bid_d + ask_d
    return float((bid_d - ask_d) / total) if total > 1e-9 else 0.0


def _safe_log(x: float) -> float:
    return float(np.log(x)) if x > 1e-9 else 0.0


def _depth_slope(bands: dict, side_bands: list) -> float:
    """
    深度曲线斜率：log(depth_far / depth_near) / pct_range
    衡量深度随距离增加的速率（越大 = 深度越集中于远档）。
    """
    d_near, _ = _get_band(bands, side_bands[0])
    d_far,  _ = _get_band(bands, side_bands[-1])
    if d_near <= 1e-9 or d_far <= 1e-9:
        return 0.0
    pct_range = abs(side_bands[-1]) - abs(side_bands[0])
    return float(np.clip(_safe_log(d_far / d_near) / (pct_range + 1e-9), -10, 10))


def compute_lob_features(
    snapshots: List[dict],
    trades: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从累计深度快照序列 + trades 离线计算 20 维 LOB 特征。

    特征定义（与 FEATURE_NAMES 对应）：
        0  spread_bps      : 买卖均价差（bps）
        1  micro_dev_bps   : 快照时间窗内净 VWAP 偏离
        2  imbalance_L1    : ±0.2% 深度不平衡
        3  imbalance_L5    : ±1.0% 深度不平衡
        4  imbalance_L10   : ±2.0% 深度不平衡
        5  imbalance_weighted: 6档指数加权不平衡
        6  depth_bid_log   : log(bid @ -5%)
        7  depth_ask_log   : log(ask @ +5%)
        8  depth_ratio     : bid/ask 总深度比
        9  top_depth_ratio : ±0.2% 深度比
       10  slope_bid_bps   : bid 侧深度曲线斜率
       11  slope_ask_bps   : ask 侧深度曲线斜率
       12  vol_decay_bid   : bid 近档集中度（±0.2% / ±5%）
       13  vol_decay_ask   : ask 近档集中度
       14  ofi_lob_100ms   : 单快照 LOB OFI（Δdepth @ ±0.2%）
       15  ofi_lob_500ms   : 近10快照累计 LOB OFI
       16  ofi_trade_100ms : 成交净流量（100ms 等效）
       17  ofi_trade_500ms : 成交净流量（500ms 等效）
       18  ofi_trade_1s    : 成交净流量（1s 等效）
       19  trade_rate_1s   : 每秒成交笔数

    参数：
        snapshots : list of {'ts_ms': int, 'bands': {pct: {depth, notional}}}
        trades    : (M, 3) [time_ms, qty, sign]，或 None

    返回：
        X      (N, 20) float32  — 特征矩阵
        ts     (N,)    int64    — 毫秒时间戳
        signs  (N,)    int8     — 下一时间窗的主导成交方向（+1/-1/0）
    """
    N          = len(snapshots)
    X          = np.zeros((N, N_LOB_FEATURES), dtype=np.float32)
    ts_out     = np.zeros(N, dtype=np.int64)
    signs_out  = np.zeros(N, dtype=np.int8)

    # trades 预处理
    if trades is not None and len(trades) > 0:
        trades   = trades[trades[:, 0].argsort()]
        n_trades = len(trades)
    else:
        trades   = np.zeros((0, 3))
        n_trades = 0

    prev_d_bid02 = None
    prev_d_ask02 = None
    ofi_lob_hist = deque()

    for i, snap in enumerate(snapshots):
        snap_ts   = snap["ts_ms"]
        bands     = snap["bands"]
        ts_out[i] = snap_ts

        if not bands:
            continue

        d_bid02, n_bid02 = _get_band(bands, -0.2)
        d_ask02, n_ask02 = _get_band(bands, +0.2)
        if d_bid02 <= 1e-9 or d_ask02 <= 1e-9:
            continue
        mid_bid = n_bid02 / d_bid02
        mid_ask = n_ask02 / d_ask02
        mid     = (mid_bid + mid_ask) / 2.0

        # 0: spread_bps
        X[i, 0] = float(np.clip(abs(mid_ask - mid_bid) / mid * 1e4, 0, 50))

        # 1: micro_dev_bps（快照后时间窗内净 qty 权重）
        next_ts = snapshots[i + 1]["ts_ms"] if i + 1 < N else snap_ts + 60000
        mask_w  = (trades[:, 0] > snap_ts) & (trades[:, 0] <= next_ts)
        if mask_w.any():
            seg = trades[mask_w]
            net_qty   = (seg[:, 1] * seg[:, 2]).sum()
            total_qty = seg[:, 1].sum()
            X[i, 1]   = float(np.clip(net_qty / (total_qty + 1e-9) * 10, -20, 20))

        # 2-5: imbalance
        d_bid1, _ = _get_band(bands, -1.0)
        d_ask1, _ = _get_band(bands, +1.0)
        d_bid2, _ = _get_band(bands, -2.0)
        d_ask2, _ = _get_band(bands, +2.0)
        d_bid5, _ = _get_band(bands, -5.0)
        d_ask5, _ = _get_band(bands, +5.0)

        X[i, 2] = _imbalance(d_bid02, d_ask02)
        X[i, 3] = _imbalance(d_bid1,  d_ask1)
        X[i, 4] = _imbalance(d_bid2,  d_ask2)

        w_arr    = np.exp(-np.arange(6) * 0.5); w_arr /= w_arr.sum()
        bid_depths = np.array([_get_band(bands, p)[0] for p in _BID_BANDS])
        ask_depths = np.array([_get_band(bands, p)[0] for p in _ASK_BANDS])
        bid_inc  = np.clip(np.diff(np.concatenate([[0], bid_depths])), 0, None)
        ask_inc  = np.clip(np.diff(np.concatenate([[0], ask_depths])), 0, None)
        bqw, aqw = (bid_inc * w_arr).sum(), (ask_inc * w_arr).sum()
        X[i, 5]  = float((bqw - aqw) / (bqw + aqw + 1e-9))

        # 6-7: depth_bid/ask_log
        X[i, 6]  = float(np.clip(_safe_log(d_bid5), -10, 15))
        X[i, 7]  = float(np.clip(_safe_log(d_ask5), -10, 15))

        # 8: depth_ratio
        X[i, 8]  = float(np.clip(d_bid5 / (d_ask5 + 1e-9), 0.001, 1000))

        # 9: top_depth_ratio
        X[i, 9]  = float(np.clip(d_bid02 / (d_ask02 + 1e-9), 0.0, 1000))

        # 10-11: slope_bid/ask
        X[i, 10] = _depth_slope(bands, _BID_BANDS)
        X[i, 11] = _depth_slope(bands, _ASK_BANDS)

        # 12-13: vol_decay（近档集中度）
        X[i, 12] = float(np.clip(d_bid02 / (d_bid5 + 1e-9), 0, 1))
        X[i, 13] = float(np.clip(d_ask02 / (d_ask5 + 1e-9), 0, 1))

        # 14-15: ofi_lob
        if prev_d_bid02 is not None:
            ofi_d  = (d_bid02 - prev_d_bid02) - (d_ask02 - prev_d_ask02)
            ofi_d /= (mid + 1e-9)
        else:
            ofi_d = 0.0
        ofi_lob_hist.append((snap_ts, ofi_d))
        while len(ofi_lob_hist) > 10:
            ofi_lob_hist.popleft()
        X[i, 14] = float(np.clip(ofi_d, -1e3, 1e3))
        X[i, 15] = float(np.clip(sum(v for _, v in ofi_lob_hist), -1e3, 1e3))

        prev_d_bid02 = d_bid02
        prev_d_ask02 = d_ask02

        # 16-19: ofi_trade + trade_rate
        prev_snap_ts = snapshots[i - 1]["ts_ms"] if i > 0 else snap_ts - 60000
        win_ms = snap_ts - prev_snap_ts
        idx_s  = np.searchsorted(trades[:, 0], prev_snap_ts, side="right")
        idx_e  = np.searchsorted(trades[:, 0], snap_ts,      side="right")
        seg_all = trades[idx_s:idx_e]
        if len(seg_all) > 0:
            net_flow   = (seg_all[:, 1] * seg_all[:, 2]).sum() / (mid + 1e-9)
            scale_100  = min(100  / win_ms, 1.0) if win_ms > 0 else 0
            scale_500  = min(500  / win_ms, 1.0) if win_ms > 0 else 0
            scale_1s   = min(1000 / win_ms, 1.0) if win_ms > 0 else 0
            X[i, 16]   = float(np.clip(net_flow * scale_100, -1e3, 1e3))
            X[i, 17]   = float(np.clip(net_flow * scale_500, -1e3, 1e3))
            X[i, 18]   = float(np.clip(net_flow * scale_1s,  -1e3, 1e3))
            trade_rate  = len(seg_all) / (win_ms / 1000.0) if win_ms > 0 else 0
            X[i, 19]   = float(np.clip(trade_rate, 0, 200))

    # 计算 signs（下一快照时间窗的净成交方向）
    if n_trades > 0:
        for i, snap in enumerate(snapshots):
            snap_ts = snap["ts_ms"]
            next_ts = snapshots[i + 1]["ts_ms"] if i + 1 < N else snap_ts + 60000
            idx_s   = np.searchsorted(trades[:, 0], snap_ts, side="right")
            idx_e   = np.searchsorted(trades[:, 0], next_ts, side="right")
            seg     = trades[idx_s:idx_e]
            if len(seg) == 0:
                signs_out[i] = 0
            else:
                signs_out[i] = int(np.sign((seg[:, 2] * seg[:, 1]).sum()))

    return X, ts_out, signs_out


# ──────────────────────────────────────────────────────────────────────────────
# 模块级工具函数：REST API LOB
# ──────────────────────────────────────────────────────────────────────────────

def fetch_depth(symbol: str, limit: int = 20,
                futures: bool = True) -> Optional[dict]:
    """
    单次拉取订单簿快照。

    参数：
        symbol  : 交易对，如 "BTCUSDT"
        limit   : 深度档位数（合约支持 5/10/20/50/100/500/1000）
        futures : True=合约（/fapi/v1/depth），False=现货（/api/v3/depth）

    返回：
        dict {'bids': [[price, qty], ...], 'asks': [...]}，失败返回 None
    """
    import json
    base = _FAPI_BASE if futures else _SPOT_BASE
    path = "/fapi/v1/depth" if futures else "/api/v3/depth"
    url  = f"{base}{path}?symbol={symbol}&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"REST depth 请求失败: {e}")
        return None


def fetch_recent_trades(symbol: str, limit: int = 100,
                        futures: bool = True) -> Optional[list]:
    """
    单次拉取最近成交。

    参数：
        symbol  : 交易对
        limit   : 最多返回 N 条（最大 1000）
        futures : True=合约，False=现货

    返回：
        list of dict {'time': ms, 'qty': str, 'isBuyerMaker': bool, ...}
        失败返回 None
    """
    import json
    base = _FAPI_BASE if futures else _SPOT_BASE
    path = "/fapi/v1/trades" if futures else "/api/v3/trades"
    url  = f"{base}{path}?symbol={symbol}&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"REST trades 请求失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# REST LOB 采集器
# ══════════════════════════════════════════════════════════════════════════════

class RestLOBCollector:
    """
    通过 REST API 轮询订单簿，持续采集 LOB 特征数据。

    采集策略：
        - 主线程：每 interval_ms 拉取一次订单簿，计算 20 维特征
        - 后台线程：每 500ms 拉取最近成交，维护成交流量滚动窗口
        - 成交特征（ofi_trade_*、trade_rate）由后台线程异步更新，主线程直接读取

    速率：
        合约 /fapi/v1/depth limit=20 消耗 2 weight，
        安全速率 4/s（每 250ms），1天 ≈ 34 万样本。

    用法：
        collector = RestLOBCollector("BTCUSDT")
        collector.collect(max_samples=200_000, output_dir="data")
    """

    def __init__(self, symbol: str, futures: bool = True,
                 interval_ms: int = 250, depth_limit: int = 20):
        self.symbol       = symbol.upper()
        self.futures      = futures
        self.interval_ms  = interval_ms
        self.depth_limit  = depth_limit

        self._X_list:     list = []
        self._ts_list:    list = []
        self._signs_list: list = []

        self._trade_wins = {
            'ofi100': deque(), 'ofi500': deque(),
            't100':   deque(), 't500':  deque(), 't1000': deque(),
        }
        self._trade_lock    = threading.Lock()
        self._last_trade_id = 0
        self._prev_bids     = None
        self._mid           = 0.0
        self._running       = False

    def _trade_worker(self) -> None:
        """
        后台线程：每 500ms 拉取最近成交，更新滚动窗口和成交特征值。

        成交特征写入 self._trade_wins['feat_*']，供主线程在下一次
        特征计算时读取（最多 500ms 延迟，对 HFT 特征来说可接受）。
        """
        while self._running:
            try:
                trades = fetch_recent_trades(self.symbol, limit=200,
                                             futures=self.futures)
                if trades:
                    mid = self._mid or 1.0
                    with self._trade_lock:
                        for t in trades:
                            tid = t.get("id", 0)
                            if tid <= self._last_trade_id:
                                continue
                            self._last_trade_id = max(self._last_trade_id, tid)
                            ts  = int(t["time"])
                            qty = float(t["qty"])
                            sgn = -1 if t.get("isBuyerMaker") else 1
                            v   = qty * sgn / mid
                            for win in [self._trade_wins['t100'],
                                        self._trade_wins['t500'],
                                        self._trade_wins['t1000']]:
                                win.append((ts, v, qty, sgn))

                        now = int(time.time() * 1000)
                        for key, maxms in [('t100', 100), ('t500', 500),
                                           ('t1000', 1000)]:
                            w = self._trade_wins[key]
                            while w and now - w[0][0] > maxms:
                                w.popleft()

                        for key in ['t100', 't500', 't1000']:
                            w   = self._trade_wins[key]
                            net = sum(v for _, v, _, _ in w)
                            self._trade_wins[f'feat_{key}'] = float(
                                np.clip(net, -1e3, 1e3))
                        self._trade_wins['feat_trate'] = float(
                            np.clip(len(self._trade_wins['t1000']), 0, 200))
            except Exception as e:
                logger.warning(f"trade_worker 错误: {e}")
            time.sleep(0.5)

    def _compute_features(self, bids: list, asks: list, ts_ms: int) -> np.ndarray:
        """
        从 REST 返回的 bids/asks 及成交窗口计算 20 维特征向量。

        直接使用前 10 档 bid/ask（而非 Binance Vision 的 6 累计深度档），
        因此部分特征的计算方式与 compute_lob_features() 略有不同，
        但语义保持一致（见 FEATURE_NAMES）。
        """
        feat = np.zeros(N_LOB_FEATURES, dtype=np.float32)
        if not bids or not asks:
            return feat

        def _parse(levels, n):
            arr = np.zeros((n, 2))
            for i, (p, q) in enumerate(levels[:n]):
                arr[i] = [float(p), float(q)]
            return arr

        b = _parse(bids, 10)
        a = _parse(asks, 10)
        best_bid, best_ask = b[0, 0], a[0, 0]
        if best_ask <= best_bid:
            return feat
        mid = (best_bid + best_ask) * 0.5

        # 0: spread_bps
        feat[0] = float(np.clip((best_ask - best_bid) / mid * 1e4, 0, 50))

        # 1: micro_dev_bps
        bv1, av1 = b[0, 1], a[0, 1]
        tv1   = bv1 + av1
        micro = (best_bid * av1 + best_ask * bv1) / tv1 if tv1 > 0 else mid
        feat[1] = float(np.clip((micro - mid) / mid * 1e4, -20, 20))

        # 2-5: imbalance（L1=1档, L5=5档, L10=10档, weighted）
        for fi, L in enumerate([1, 5, 10, 10]):
            bq = b[:L, 1].sum(); aq = a[:L, 1].sum()
            tot = bq + aq
            imb = (bq - aq) / tot if tot > 0 else 0.0
            if fi == 3:
                w = np.exp(-np.arange(10) * 0.3); w /= w.sum()
                bqw = (b[:, 1] * w).sum(); aqw = (a[:, 1] * w).sum()
                tw  = bqw + aqw
                imb = (bqw - aqw) / tw if tw > 0 else 0.0
            feat[2 + fi] = float(np.clip(imb, -1, 1))

        # 6-7: depth log（全10档总量）
        feat[6] = float(np.clip(np.log(b[:, 1].sum() + 1e-9), -10, 10))
        feat[7] = float(np.clip(np.log(a[:, 1].sum() + 1e-9), -10, 10))

        # 8: depth_ratio
        feat[8] = float(np.clip(b[:, 1].sum() / (a[:, 1].sum() + 1e-9),
                                0.001, 1000))

        # 9: top_depth_ratio
        feat[9] = float(np.clip(bv1 / (av1 + 1e-9), 0, 1e4))

        # 10-11: slope_bid/ask（量随价格档位的变化率）
        for si, lvl in enumerate([b, a]):
            p0 = lvl[0, 0]; pL = lvl[min(4, len(lvl) - 1), 0]
            dp = abs(pL - p0) / mid * 1e4
            dq = lvl[min(4, len(lvl) - 1), 1] - lvl[0, 1]
            feat[10 + si] = float(np.clip(abs(dq / dp) if dp > 1e-9 else 0,
                                          0, 500))

        # 12-13: vol_decay（指数加权集中度）
        for si, lvl in enumerate([b, a]):
            tot = lvl[:, 1].sum()
            if tot > 0:
                w = np.exp(-np.arange(10) * 0.5); w /= w.sum()
                feat[12 + si] = float(np.clip(
                    (lvl[:, 1] / tot * w).sum() * 100, 0, 10))

        # 14-15: ofi_lob（与前一快照的 best bid/ask 量变化）
        with self._trade_lock:
            if self._prev_bids is not None:
                pb  = _parse(self._prev_bids, 1)
                db  = (bv1 if best_bid >= pb[0, 0] else 0) - \
                      (bv1 if best_bid <= pb[0, 0] else 0)
                da  = (av1 if best_ask <= a[0, 0] else 0) - \
                      (av1 if best_ask >= a[0, 0] else 0)
                ofi = (db - da) / (mid + 1e-9)
                win100 = self._trade_wins['ofi100']
                win500 = self._trade_wins['ofi500']
                win100.append((ts_ms, ofi)); win500.append((ts_ms, ofi))
                while win100 and ts_ms - win100[0][0] > 100: win100.popleft()
                while win500 and ts_ms - win500[0][0] > 500: win500.popleft()
                feat[14] = float(np.clip(sum(v for _, v in win100), -1e3, 1e3))
                feat[15] = float(np.clip(sum(v for _, v in win500), -1e3, 1e3))

            # 16-19: trade OFI（由后台线程异步维护）
            for fi, key in enumerate(['feat_t100', 'feat_t500',
                                      'feat_t1000', 'feat_trate']):
                feat[16 + fi] = self._trade_wins.get(key, 0.0)

        return feat

    def collect(self, max_samples: int = 100_000,
                duration_s: float = None,
                output_dir: str = "data") -> None:
        """
        持续采集 LOB 特征并保存到 output_dir。

        参数：
            max_samples : 最大样本数，到达后自动停止
            duration_s  : 最大采集时长（秒），与 max_samples 取先到者
            output_dir  : .npy 文件保存目录

        保存文件：
            {symbol}_lob_X.npy     (N, 20) float32
            {symbol}_lob_ts.npy    (N,)    int64
            {symbol}_lob_signs.npy (N,)    int8   （全0，需后处理填充方向标签）
        """
        os.makedirs(output_dir, exist_ok=True)
        self._running = True

        t_thread = threading.Thread(target=self._trade_worker, daemon=True)
        t_thread.start()

        t0         = time.time()
        last_print = t0
        count      = 0

        print(f"\n[{self.symbol}] REST LOB 采集 (间隔={self.interval_ms}ms)")
        print(f"  目标: {max_samples:,} 样本  按 Ctrl+C 停止\n")

        try:
            while True:
                loop_start = time.time()

                if count >= max_samples:
                    break
                if duration_s and (loop_start - t0) >= duration_s:
                    break

                depth   = fetch_depth(self.symbol, self.depth_limit, self.futures)
                snap_ts = int(time.time() * 1000)

                if depth and depth.get("bids") and depth.get("asks"):
                    bids     = depth["bids"]
                    asks     = depth["asks"]
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    self._mid = (best_bid + best_ask) / 2.0

                    feat = self._compute_features(bids, asks, snap_ts)

                    self._prev_bids = bids
                    self._X_list.append(feat)
                    self._ts_list.append(snap_ts)
                    self._signs_list.append(0)
                    count += 1

                if time.time() - last_print >= 5.0:
                    elapsed  = time.time() - t0
                    rate     = count / max(elapsed, 1)
                    eta_min  = (max_samples - count) / max(rate, 0.1) / 60
                    print(f"  已采集: {count:>8,}  速率: {rate:>6.1f}/s  "
                          f"ETA: {eta_min:.1f}min", flush=True)
                    last_print = time.time()

                sleep = self.interval_ms / 1000.0 - (time.time() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            print("\n  用户中断，保存数据...")

        self._running = False
        self._save(output_dir)

    def _save(self, output_dir: str) -> None:
        if not self._X_list:
            print("  无数据，跳过保存。")
            return
        X  = np.array(self._X_list,     dtype=np.float32)
        ts = np.array(self._ts_list,    dtype=np.int64)
        sg = np.array(self._signs_list, dtype=np.int8)

        sym = self.symbol
        np.save(os.path.join(output_dir, f"{sym}_lob_X.npy"),     X)
        np.save(os.path.join(output_dir, f"{sym}_lob_ts.npy"),    ts)
        np.save(os.path.join(output_dir, f"{sym}_lob_signs.npy"), sg)

        rate = len(X) / max((ts[-1] - ts[0]) / 1000, 1)
        print(f"""
╔══════════════════════════════════════════════════╗
  REST LOB 采集完成
  样本数 : {len(X):,}
  时间跨度: {(ts[-1]-ts[0])/1000:.0f}s
  平均速率: {rate:.1f}/s

  X     → {output_dir}/{sym}_lob_X.npy
  ts    → {output_dir}/{sym}_lob_ts.npy
  signs → {output_dir}/{sym}_lob_signs.npy  (全0，需后处理)

  训练:
    python analysis/train_manifold.py \\
      --X {output_dir}/{sym}_lob_X.npy \\
      --signs {output_dir}/{sym}_lob_signs.npy
╚══════════════════════════════════════════════════╝""")


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader 主类
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    统一数据加载与采集工具类。

    封装所有数据源的下载、解析和加载逻辑，提供标准格式的输出：
        - K 线 / 技术指标     : pandas DataFrame
        - LOB 特征矩阵        : numpy ndarray
        - Hawkes 事件序列     : list of (t_sec, event_type)

    数据目录约定（self.data_dir）：
        - 所有下载文件默认存入此目录，文件名含品种/日期/时间范围
        - 避免重复下载（文件存在时自动跳过）
    """

    def __init__(self, data_dir: str = "data"):
        """
        参数：
            data_dir : 本地数据存储目录（不存在时自动创建）
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # K 线 / 行情
    # ──────────────────────────────────────────────────────────────────────────

    def load_from_csv(self, filepath: str, symbol: str = "ETHUSDT") -> pd.DataFrame:
        """
        从 CSV 文件加载历史 K 线数据，并标准化为系统统一格式。

        为什么需要列名标准化？
            不同数据源（TradingView 导出、Binance 导出、第三方数据商）
            使用不同的列名约定（如 "timestamp" vs "time" vs "datetime"）。
            统一列名后，下游代码只需处理一种格式，提高可复用性。

        参数：
            filepath (str) : CSV 文件路径
            symbol   (str) : 交易对名称（当前未使用，预留供将来过滤多品种 CSV 用）

        返回：
            pd.DataFrame : 标准格式的 OHLCV 数据，以 datetime 为索引
        """
        try:
            df = pd.read_csv(filepath)

            column_mapping = {
                'timestamp': 'datetime',
                'time': 'datetime',
                'open_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'close_price': 'close',
                'volume_traded': 'volume'
            }
            df = df.rename(columns=column_mapping)

            if 'datetime' not in df.columns:
                if 'date' in df.columns and 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                elif 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'])
                else:
                    df['datetime'] = pd.date_range(
                        start='2020-01-01', periods=len(df), freq='1min')

            df = df.set_index('datetime')

            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = df.get('close', 100)

            logger.info(f"从CSV加载数据成功: {filepath}, 数据量: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"加载CSV数据失败: {e}")
            raise

    def download_binance_data(self, symbol: str, interval: str = "15m",
                              start_date: str = "2020-01-01",
                              end_date: str = None) -> pd.DataFrame:
        """
        从 Binance REST API 下载指定品种和时间范围的历史 K 线数据。

        为什么要分页下载？
            Binance /api/v3/klines 单次请求最多返回 1000 根 K 线。
            若需要更长的历史数据，必须以最后一根 K 线的时间戳为起点
            发起新的请求，循环直到覆盖完整时间范围。

        参数：
            symbol     (str) : 交易对，例如 "BTCUSDT"
            interval   (str) : K 线周期，Binance 支持 1m/5m/15m/1h/4h/1d 等
            start_date (str) : 开始日期，格式 "YYYY-MM-DD"
            end_date   (str) : 结束日期，格式 "YYYY-MM-DD"；None 表示到今天

        返回：
            pd.DataFrame : 标准格式的 OHLCV 数据，已保存到本地 CSV
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            base_url   = "https://api.binance.com/api/v3/klines"
            start_ts   = int(pd.to_datetime(start_date).timestamp() * 1000)
            end_ts     = int(pd.to_datetime(end_date).timestamp() * 1000)
            all_data   = []
            current_start = start_ts

            while current_start < end_ts:
                params = {
                    'symbol':    symbol,
                    'interval':  interval,
                    'startTime': current_start,
                    'endTime':   end_ts,
                    'limit':     1000,
                }
                response = requests.get(base_url, params=params)
                data     = response.json()
                if not data:
                    break
                all_data.extend(data)
                current_start = data[-1][0] + 1

            df = pd.DataFrame(all_data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                'ignore'
            ])
            df['open_time']  = pd.to_datetime(df['open_time'],  unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df = df.set_index('open_time')
            df = df.rename(columns={'open_time': 'datetime'})

            filename = (f"{symbol}_{interval}_{start_date.replace('-', '')}"
                        f"_{end_date.replace('-', '')}.csv")
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath)

            logger.info(f"下载Binance数据成功: {symbol}, 数据量: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"下载Binance数据失败: {e}")
            raise

    def download_agg_trades(self, symbol: str, hours: float,
                             save: bool = True) -> pd.DataFrame:
        """
        从 Binance 下载最近 N 小时的聚合逐笔成交数据（aggTrades），
        并可选地保存为本地 CSV（默认开启，文件名含品种和时间范围）。

        参数：
            symbol (str)   : 交易对，例如 "BTCUSDT"
            hours  (float) : 下载最近 N 小时的数据
            save   (bool)  : 是否将结果保存到 data_dir/CSV，默认 True

        返回：
            pd.DataFrame : 列 [timestamp(ms), price, qty, is_buyer_maker]
        """
        end_ms   = int(time.time() * 1000)
        start_ms = int(end_ms - hours * 3_600_000)

        url        = "https://api.binance.com/api/v3/aggTrades"
        all_trades = []
        current    = start_ms

        logger.info(f"[DataLoader] 下载 {symbol} 最近 {hours}h aggTrades...")

        while current < end_ms:
            resp = requests.get(url, params={
                "symbol":    symbol,
                "startTime": current,
                "endTime":   end_ms,
                "limit":     1000,
            }, timeout=10)
            resp.raise_for_status()

            batch = resp.json()
            if not batch:
                break

            all_trades.extend(batch)
            current = batch[-1]["T"] + 1

            pct = min(100, (current - start_ms) / (end_ms - start_ms) * 100)
            print(f"\r[数据下载] {len(all_trades)} 条  {pct:.0f}%",
                  end="", flush=True)

        print()

        if not all_trades:
            raise RuntimeError(
                f"未获取到任何 Tick 数据，请检查网络或 symbol={symbol}")

        df = pd.DataFrame([{
            "timestamp":      t["T"],
            "price":          float(t["p"]),
            "qty":            float(t["q"]),
            "is_buyer_maker": bool(t["m"]),
        } for t in all_trades])

        logger.info(
            f"[DataLoader] 下载完成 {len(df)} 条  "
            f"{pd.to_datetime(df['timestamp'].iloc[0],  unit='ms')} ~ "
            f"{pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}"
        )

        if save:
            ts_start = pd.to_datetime(
                df["timestamp"].iloc[0],  unit="ms").strftime("%Y%m%d_%H%M")
            ts_end   = pd.to_datetime(
                df["timestamp"].iloc[-1], unit="ms").strftime("%Y%m%d_%H%M")
            filename = f"{symbol}_tick_{ts_start}_{ts_end}.csv"
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"[DataLoader] Tick 数据已保存 → {filepath}")

        return df

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        将 K 线数据重采样（聚合）到更长的时间周期。

        OHLC 聚合规则（业界标准）：
            open=first, high=max, low=min, close=last, volume=sum

        参数：
            df        (DataFrame) : 原始 K 线数据（以 datetime 为索引）
            timeframe (str)       : 目标时间周期，支持 '1m'/'5m'/'15m'/'1h'/'4h'/'1d'

        返回：
            pd.DataFrame : 重采样后的 K 线数据
        """
        timeframe_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '1h': '1H',   '4h': '4H',   '1d': '1D'
        }
        if timeframe not in timeframe_map:
            raise ValueError(f"不支持的时间周期: {timeframe}")

        ohlc_dict = {
            'open': 'first', 'high': 'max',
            'low':  'min',   'close': 'last', 'volume': 'sum'
        }
        resampled_df = df.resample(timeframe_map[timeframe]).agg(ohlc_dict).dropna()
        logger.info(f"数据重采样完成: {timeframe}, 新数据量: {len(resampled_df)}")
        return resampled_df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加常用技术指标（MA/RSI/MACD/布林带）。

        使用 df.copy() 避免修改原始数据；所有指标作为新列追加。

        返回：
            pd.DataFrame : 添加了技术指标列的新 DataFrame
        """
        df = df.copy()

        df['ma_5']  = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_60'] = df['close'].rolling(window=60).mean()

        delta = df['close'].diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd']           = exp1 - exp2
        df['macd_signal']    = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std          = df['close'].rolling(window=20).std()
        df['bb_upper']  = df['bb_middle'] + (bb_std * 2)
        df['bb_lower']  = df['bb_middle'] - (bb_std * 2)

        logger.info("技术指标添加完成")
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证 DataFrame 数据质量（空值、异常价格、时间单调性）。

        返回：
            bool : True=通过，False=存在质量问题
        """
        if df.isnull().any().any():
            logger.warning("数据中存在空值")
            return False
        if (df['close'] <= 0).any():
            logger.warning("价格数据异常")
            return False
        time_diff = df.index.to_series().diff().dropna()
        if time_diff.min() < pd.Timedelta(0):
            logger.warning("时间数据不连续")
            return False
        logger.info("数据验证通过")
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Binance Vision 历史 LOB 下载
    # ──────────────────────────────────────────────────────────────────────────

    def download_lob_history(
        self,
        symbol: str,
        dates: List[str] = None,
        days: int = 1,
        output_dir: str = None,
        keep_zip: bool = False,
        book_zip: str = None,
        trades_zip: str = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从 Binance Vision 下载历史 bookDepth+trades 并计算 LOB 特征。

        优先级：book_zip（单文件） > dates（指定日期） > days（最近N天）

        参数：
            symbol      : 交易对，如 "BTCUSDT"
            dates       : 指定日期列表，格式 ["YYYY-MM-DD", ...]
            days        : 从最新可用日期往前 N 天（dates 为 None 时生效）
            output_dir  : .npy 和 .zip 文件保存目录（默认 self.data_dir）
            keep_zip    : 是否保留下载的 zip 文件（默认 False）
            book_zip    : 已有本地 bookDepth zip 路径（跳过下载）
            trades_zip  : 已有本地 trades zip 路径（跳过下载）

        返回：
            X      (N, 20) float32  — LOB 特征矩阵
            ts     (N,)    int64    — 毫秒时间戳
            signs  (N,)    int8     — 主导成交方向

        同时将结果保存为：
            {output_dir}/{symbol}_lob_X.npy
            {output_dir}/{symbol}_lob_ts.npy
            {output_dir}/{symbol}_lob_signs.npy
        """
        if output_dir is None:
            output_dir = self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        symbol = symbol.upper()

        def _process_day(sym, date_str, b_zip=None, t_zip=None):
            if b_zip is None:
                b_zip = os.path.join(output_dir,
                                     f"{sym}-bookDepth-{date_str}.zip")
                download_file(_url_bookdepth(sym, date_str),
                              b_zip, required=True)
            if t_zip is None:
                t_zip = os.path.join(output_dir,
                                     f"{sym}-trades-{date_str}.zip")
                download_file(_url_trades(sym, date_str),
                              t_zip, required=False)

            logger.info(f"[{date_str}] 解析 bookDepth...")
            snapshots = parse_bookdepth_zip(b_zip)

            trades = None
            if os.path.exists(t_zip):
                logger.info(f"[{date_str}] 解析 trades...")
                trades = parse_trades_zip(t_zip)

            logger.info(f"[{date_str}] 计算 LOB 特征...")
            t0 = time.time()
            X, ts, signs = compute_lob_features(snapshots, trades)
            logger.info(f"[{date_str}] 完成 N={len(X)}  "
                        f"耗时={time.time()-t0:.1f}s")

            if not keep_zip:
                for p in [b_zip, t_zip]:
                    if p and os.path.exists(p):
                        os.remove(p)
            return X, ts, signs

        # ── 单文件模式 ───────────────────────────────────────────────────────
        if book_zip:
            X, ts, signs = _process_day(symbol, "manual", book_zip, trades_zip)
            dates_processed = ["manual"]
        else:
            # ── 确定日期列表 ─────────────────────────────────────────────────
            if dates:
                date_list = dates
            else:
                print(f"  探测 Binance Vision 最新可用日期...")
                latest = probe_latest_date(symbol)
                if latest is None:
                    raise RuntimeError(
                        "无法找到最近 7 天内的可用数据，请手动指定 dates。")
                latest_date = datetime.strptime(latest, "%Y-%m-%d").date()
                date_list = [
                    (latest_date - timedelta(days=d)).strftime("%Y-%m-%d")
                    for d in range(days)
                ]
                print(f"  最新可用: {latest}，下载 {len(date_list)} 天: {date_list}")

            all_X_list:    list = []
            all_ts_list:   list = []
            all_signs_list: list = []

            for d in date_list:
                print(f"\n─── {d} ──────────────────────────────────────────────")
                try:
                    Xd, tsd, sd = _process_day(symbol, d)
                    all_X_list.append(Xd)
                    all_ts_list.append(tsd)
                    all_signs_list.append(sd)
                except Exception as e:
                    logger.error(f"  {d} 处理失败: {e}")

            if not all_X_list:
                raise RuntimeError("没有成功处理任何数据。")

            X      = np.vstack(all_X_list)
            ts     = np.concatenate(all_ts_list)
            signs  = np.concatenate(all_signs_list)
            dates_processed = date_list

        # ── 保存 ─────────────────────────────────────────────────────────────
        x_path     = os.path.join(output_dir, f"{symbol}_lob_X.npy")
        ts_path    = os.path.join(output_dir, f"{symbol}_lob_ts.npy")
        signs_path = os.path.join(output_dir, f"{symbol}_lob_signs.npy")

        np.save(x_path,     X.astype(np.float32))
        np.save(ts_path,    ts.astype(np.int64))
        np.save(signs_path, signs.astype(np.int8))

        buy_rate = (signs == 1).mean()
        print(f"""
╔══════════════════════════════════════════════════╗
  LOB 历史数据下载 + 特征计算完成
  日期  : {', '.join(str(d) for d in dates_processed[:3])}\
{'...' if len(dates_processed) > 3 else ''}
  样本数: {len(X):,}
  特征  : {X.shape[1]} 维
  主动买: {buy_rate:.1%}

  X     → {x_path}
  ts    → {ts_path}
  signs → {signs_path}

  运行训练:
    python analysis/train_manifold.py \\
      --X {x_path} \\
      --signs {signs_path}
╚══════════════════════════════════════════════════╝""")

        return X, ts, signs

    # ──────────────────────────────────────────────────────────────────────────
    # REST API 实时 LOB 采集
    # ──────────────────────────────────────────────────────────────────────────

    def collect_rest_lob(
        self,
        symbol: str,
        max_samples: int = 200_000,
        duration_s: float = None,
        interval_ms: int = 250,
        futures: bool = True,
        output_dir: str = None,
    ) -> None:
        """
        通过 REST API 轮询采集实时 LOB 特征（封装 RestLOBCollector）。

        适用于：没有 WebSocket 权限 / 需要精确控制采集频率的场景。

        参数：
            symbol      : 交易对
            max_samples : 最大样本数
            duration_s  : 最大采集时长（秒），与 max_samples 取先到者
            interval_ms : 轮询间隔毫秒（默认 250ms = 4/s）
            futures     : True=合约，False=现货
            output_dir  : 保存目录（默认 self.data_dir）
        """
        if output_dir is None:
            output_dir = self.data_dir
        collector = RestLOBCollector(
            symbol=symbol,
            futures=futures,
            interval_ms=interval_ms,
        )
        collector.collect(
            max_samples=max_samples,
            duration_s=duration_s,
            output_dir=output_dir,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Hawkes 训练数据加载
    # ──────────────────────────────────────────────────────────────────────────

    def load_trades_csv(self, path: str) -> list:
        """
        加载单个 trades CSV，返回 Hawkes 事件列表。

        自动识别两种格式：
            Binance Vision  : id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
            REST API 采集   : timestamp, side, price, qty, is_buyer_maker, ...

        返回：
            events : list of (t_sec, event_type)
                event_type: 0=主动买（not isBuyerMaker），1=主动卖（isBuyerMaker）
        """
        try:
            import pandas as _pd
        except ImportError:
            raise ImportError("需要 pandas: pip install pandas")

        df = _pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]

        if "isBuyerMaker" in df.columns and "time" in df.columns:
            df["isBuyerMaker"] = df["isBuyerMaker"].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False})
        elif "is_buyer_maker" in df.columns and "timestamp" in df.columns:
            df["isBuyerMaker"] = df["is_buyer_maker"].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False})
            df = df.rename(columns={"timestamp": "time"})
        else:
            raise ValueError(
                f"无法识别 trades CSV 格式，列名: {list(df.columns)}")

        events = [
            (row["time"] / 1000.0,
             1 if row["isBuyerMaker"] else 0)
            for _, row in df.iterrows()
        ]
        logger.info(f"[DataLoader] 加载 {os.path.basename(path)}: "
                    f"{len(events)} 个 Hawkes 事件")
        return events

    def load_all_trades(self, pattern_or_paths) -> list:
        """
        加载多个 trades CSV 并合并为 Hawkes 事件序列（按时间排序）。

        参数：
            pattern_or_paths : str（glob 模式，如 "data/BTCUSDT_tick_*.csv"）
                               或 list of str（文件路径列表）

        返回：
            events : list of (t_sec, event_type)，时间已归零（t0=0）

        归零原因：Hawkes MLE 与绝对时间无关，归零后数值更稳定。
        """
        if isinstance(pattern_or_paths, str):
            paths = sorted(_glob_module.glob(pattern_or_paths))
        else:
            paths = sorted(pattern_or_paths)

        if not paths:
            raise FileNotFoundError(f"未找到任何文件: {pattern_or_paths}")

        all_events = []
        for p in paths:
            evs = self.load_trades_csv(p)
            all_events.extend(evs)

        all_events.sort(key=lambda x: x[0])

        if all_events:
            t0 = all_events[0][0]
            all_events = [(t - t0, ev) for t, ev in all_events]

        logger.info(f"[DataLoader] 合并 {len(paths)} 个文件，"
                    f"共 {len(all_events)} 个事件")
        return all_events

    # ──────────────────────────────────────────────────────────────────────────
    # 特征矩阵加载
    # ──────────────────────────────────────────────────────────────────────────

    def load_lob_features(
        self,
        x_path: str,
        signs_path: str = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        加载 .npy 格式的 LOB 特征矩阵和方向标签，并打印统计信息。

        参数：
            x_path     : 特征矩阵路径（{symbol}_lob_X.npy）
            signs_path : 方向标签路径（{symbol}_lob_signs.npy），None 则跳过
            verbose    : 是否打印详细统计（默认 True）

        返回：
            X      (N, 20) float32  — 已替换 NaN/Inf 为 0
            signs  (N,)    int8 或 None
        """
        X = np.load(x_path)

        if verbose:
            print(f"\n  特征矩阵  : {x_path}")
            print(f"  shape     : {X.shape}   dtype={X.dtype}")

        nan_cnt = np.isnan(X).sum()
        inf_cnt = np.isinf(X).sum()
        if nan_cnt + inf_cnt > 0:
            if verbose:
                print(f"  ⚠ NaN={nan_cnt}  Inf={inf_cnt}，已替换为 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        elif verbose:
            print(f"  NaN={nan_cnt}  Inf={inf_cnt}  ✓")

        if verbose:
            print(f"\n  {'特征名':<22} {'均值':>9} {'标准差':>9} "
                  f"{'最小值':>9} {'最大值':>9}")
            print(f"  {'─'*62}")
            for i, name in enumerate(FEATURE_NAMES):
                col = X[:, i]
                print(f"  {name:<22} {col.mean():9.3f} {col.std():9.3f} "
                      f"{col.min():9.3f} {col.max():9.3f}")

        signs = None
        if signs_path and os.path.exists(signs_path):
            signs    = np.load(signs_path)
            signs    = signs[:len(X)]
            buy_rate = (signs == 1).mean()
            if verbose:
                print(f"\n  成交方向  : {signs_path}")
                print(f"  shape     : {signs.shape}   "
                      f"主动买占比={buy_rate:.3f}")
        elif verbose:
            print("\n  成交方向  : 未提供")

        logger.info(f"[DataLoader] 特征矩阵加载完成: {X.shape}")
        return X, signs

    # ──────────────────────────────────────────────────────────────────────────
    # Tardis.dev 数据接入
    # ──────────────────────────────────────────────────────────────────────────

    def get_tardis_loader(
        self,
        api_key:  str = "",
        exchange: str = "binance-futures",
    ):
        """
        返回 TardisDataLoader 实例，与当前 DataLoader 共享 data_dir。

        用法：
            loader = DataLoader(data_dir="data")
            tardis = loader.get_tardis_loader(api_key="YOUR_KEY")

            # 下载并计算 LOB 特征
            X, ts, signs = tardis.compute_lob_features(
                "BTCUSDT", "2024-01-01", "2024-01-03")

            # 加载 Hawkes 训练数据
            events = tardis.load_trades_for_hawkes(
                "BTCUSDT", "2024-01-01", "2024-01-03")

        参数：
            api_key  : Tardis API Key（在 tardis.dev 注册后获取）
            exchange : 交易所标识符，默认 "binance-futures"

        返回：
            TardisDataLoader 实例，download_dir="{data_dir}/tardis"
        """
        from data_layer.tardis_loader import TardisDataLoader
        return TardisDataLoader(
            api_key      = api_key,
            exchange     = exchange,
            download_dir = os.path.join(self.data_dir, "tardis"),
        )

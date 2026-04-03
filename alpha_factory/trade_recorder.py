"""
TradeRecorder - 交易轨迹记录与平仓归因模块

设计目标：
    1. 每个持仓在内存中维护一条事件轨迹（TradeJournal），记录从开仓到平仓的关键事件
    2. 平仓时将轨迹一次性导出为独立 CSV（按 trade_id 命名，位于 trades/ 子目录）
    3. 维护三张全局追加表：
        trades.csv          — 交易摘要（每笔平仓一行）
        lifecycle_events.csv — 所有 lifecycle 状态变化记录
        exit_log.csv        — 平仓决策完整快照（含 lifecycle 上下文）

事件类型（event_type 字段）：
    entry              — 开仓
    lifecycle_change   — lifecycle 状态机转换（BUILD/EXPANSION/DECAY/REVERSAL）
    rank_snapshot      — 每轮排序时的 alpha score 快照
    trailing_armed     — trailing 激活（开始盈利超过 trailing_min_profit）
    trailing_hit       — trailing 触发平仓前的最终状态快照
    peak_pnl_update    — 峰值盈利更新
    exit               — 平仓（exit_reason + 完整上下文）
"""

import csv
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── 事件类型常量 ──────────────────────────────────────────────────────────────

EVENT_ENTRY            = "entry"
EVENT_LIFECYCLE_CHANGE = "lifecycle_change"
EVENT_RANK_SNAPSHOT    = "rank_snapshot"
EVENT_TRAILING_ARMED   = "trailing_armed"
EVENT_TRAILING_HIT     = "trailing_hit"
EVENT_PEAK_PNL_UPDATE  = "peak_pnl_update"
EVENT_EXIT             = "exit"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class TradeEvent:
    """单条轨迹事件"""
    ts:           float          # unix timestamp
    event_type:   str            # EVENT_* 常量
    price:        float = 0.0
    # lifecycle 上下文
    lc_state:     str   = ""     # BUILD / EXPANSION / DECAY / REVERSAL
    aligned_score: float = 0.0   # 方向对齐的 unified score（LONG=score, SHORT=-score）
    velocity:     float = 0.0    # lifecycle velocity EWMA
    # 盈亏上下文
    cur_ret:      float = 0.0    # 当前收益率（价格维度，未杠杆）
    peak_pnl:     float = 0.0    # 持仓期间峰值 USDT PnL
    drawdown:     float = 0.0    # 从峰值价格的回撤比例
    # 退出专用字段
    exit_reason:  str   = ""
    # 其他上下文（rank_snapshot 用）
    extra:        str   = ""     # 自由文本，如 rank_score / timing_score


@dataclass
class TradeJournal:
    """单笔持仓的完整事件轨迹"""
    trade_id:    str
    symbol:      str
    side:        str    # LONG / SHORT
    entry_price: float
    entry_time:  float
    qty:         float
    leverage:    int
    open_vol:    float  # 入场时波动率（Layer 1 硬止损基准）
    entry_score: float  # 开仓时 unified alpha
    events:      List[TradeEvent] = field(default_factory=list)

    def add(self, event: TradeEvent):
        self.events.append(event)


# ── 主模块 ────────────────────────────────────────────────────────────────────

class TradeRecorder:
    """
    使用方式：
        recorder = TradeRecorder(export_dir=".")

        # 开仓时
        trade_id = recorder.open(symbol, side, price, qty, leverage, open_vol, entry_score)

        # 关键事件时（lifecycle 变化、rank 快照、trailing 激活等）
        recorder.record_event(trade_id, TradeEvent(...))

        # 平仓时
        recorder.close(trade_id, exit_price, exit_reason, lc_state, aligned_score,
                       velocity, peak_pnl, drawdown, cur_ret, pnl_usdt, fee_usdt,
                       ret_pct, ret_lev_pct, hold_seconds)
    """

    # ── 全局表的 CSV 列定义 ──────────────────────────────────────────────────

    _TRADES_COLS = [
        "trade_id", "symbol", "side", "entry_time", "exit_time",
        "entry_price", "exit_price", "qty", "leverage",
        "open_vol", "entry_score",
        "exit_reason", "hold_seconds",
        "ret_pct", "ret_lev_pct", "pnl_usdt", "fee_usdt", "net_pnl",
        "lc_state_at_exit", "aligned_score_at_exit", "velocity_at_exit",
        "peak_pnl", "drawdown_at_exit",
    ]

    _LC_COLS = [
        "ts", "trade_id", "symbol", "side",
        "old_state", "new_state",
        "aligned_score", "velocity", "cur_ret", "price",
    ]

    _EXIT_COLS = [
        "ts", "trade_id", "symbol", "side",
        "exit_reason",
        "lc_state", "aligned_score", "velocity",
        "cur_ret", "peak_pnl", "drawdown",
        "entry_price", "exit_price", "hold_seconds",
        "pnl_usdt", "fee_usdt", "net_pnl",
        "ret_pct", "ret_lev_pct",
    ]

    def __init__(self, export_dir: str = ".", reset: bool = False):
        self._dir       = export_dir
        self._trail_dir = os.path.join(export_dir, "trades")
        os.makedirs(self._trail_dir, exist_ok=True)

        self._journals: Dict[str, TradeJournal] = {}  # trade_id → journal
        # symbol+side → trade_id（方便策略层按持仓 key 查询）
        self._active: Dict[tuple, str] = {}

        # 全局表路径
        self._trades_path = os.path.join(export_dir, "trades.csv")
        self._lc_path     = os.path.join(export_dir, "lifecycle_events.csv")
        self._exit_path   = os.path.join(export_dir, "exit_log.csv")

        # reset=True：强制清空三张全局表（每次启动重新开始）
        # reset=False：文件存在则追加，不存在才写表头
        if reset:
            self._reset_csv(self._trades_path, self._TRADES_COLS)
            self._reset_csv(self._lc_path,     self._LC_COLS)
            self._reset_csv(self._exit_path,   self._EXIT_COLS)
        else:
            self._init_csv(self._trades_path, self._TRADES_COLS)
            self._init_csv(self._lc_path,     self._LC_COLS)
            self._init_csv(self._exit_path,   self._EXIT_COLS)

    # ─── 公开接口 ─────────────────────────────────────────────────────────────

    def open(
        self,
        symbol:      str,
        side:        str,
        price:       float,
        qty:         float,
        leverage:    int,
        open_vol:    float,
        entry_score: float,
    ) -> str:
        """注册新持仓，返回 trade_id"""
        trade_id = f"{symbol}_{side}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        journal  = TradeJournal(
            trade_id    = trade_id,
            symbol      = symbol,
            side        = side,
            entry_price = price,
            entry_time  = time.time(),
            qty         = qty,
            leverage    = leverage,
            open_vol    = open_vol,
            entry_score = entry_score,
        )
        journal.add(TradeEvent(
            ts         = journal.entry_time,
            event_type = EVENT_ENTRY,
            price      = price,
            extra      = f"score={entry_score:+.3f} vol={open_vol:.4%}",
        ))
        self._journals[trade_id] = journal
        self._active[(symbol, side)] = trade_id
        return trade_id

    def get_trade_id(self, symbol: str, side: str) -> Optional[str]:
        return self._active.get((symbol, side))

    def record_lifecycle_change(
        self,
        trade_id:     str,
        old_state:    str,
        new_state:    str,
        aligned_score: float,
        velocity:     float,
        cur_ret:      float,
        price:        float,
    ):
        """记录 lifecycle 状态变化，同时追加到全局 lifecycle_events.csv"""
        j = self._journals.get(trade_id)
        if j is None:
            return
        now = time.time()
        j.add(TradeEvent(
            ts            = now,
            event_type    = EVENT_LIFECYCLE_CHANGE,
            price         = price,
            lc_state      = new_state,
            aligned_score = aligned_score,
            velocity      = velocity,
            cur_ret       = cur_ret,
            extra         = f"{old_state}->{new_state}",
        ))
        self._append_csv(self._lc_path, self._LC_COLS, {
            "ts":            round(now, 3),
            "trade_id":      trade_id,
            "symbol":        j.symbol,
            "side":          j.side,
            "old_state":     old_state,
            "new_state":     new_state,
            "aligned_score": round(aligned_score, 4),
            "velocity":      round(velocity, 5),
            "cur_ret":       round(cur_ret, 5),
            "price":         round(price, 6),
        })

    def record_rank_snapshot(
        self,
        trade_id:    str,
        aligned_score: float,
        velocity:    float,
        lc_state:    str,
        price:       float,
        extra:       str = "",
    ):
        """每轮排序时记录 alpha score 快照"""
        j = self._journals.get(trade_id)
        if j is None:
            return
        j.add(TradeEvent(
            ts            = time.time(),
            event_type    = EVENT_RANK_SNAPSHOT,
            price         = price,
            lc_state      = lc_state,
            aligned_score = aligned_score,
            velocity      = velocity,
            extra         = extra,
        ))

    def record_trailing_armed(
        self,
        trade_id: str,
        price:    float,
        cur_ret:  float,
        lc_state: str,
    ):
        """trailing 激活时记录（盈利首次超过 trailing_min_profit）"""
        j = self._journals.get(trade_id)
        if j is None:
            return
        j.add(TradeEvent(
            ts         = time.time(),
            event_type = EVENT_TRAILING_ARMED,
            price      = price,
            lc_state   = lc_state,
            cur_ret    = cur_ret,
        ))

    def record_trailing_hit(
        self,
        trade_id:  str,
        price:     float,
        cur_ret:   float,
        drawdown:  float,
        lc_state:  str,
        threshold: float,
    ):
        """trailing 即将触发时的状态快照"""
        j = self._journals.get(trade_id)
        if j is None:
            return
        j.add(TradeEvent(
            ts         = time.time(),
            event_type = EVENT_TRAILING_HIT,
            price      = price,
            lc_state   = lc_state,
            cur_ret    = cur_ret,
            drawdown   = drawdown,
            extra      = f"thresh={threshold:.4%}",
        ))

    def record_peak_pnl(
        self,
        trade_id: str,
        price:    float,
        peak_pnl: float,
        cur_ret:  float,
    ):
        """峰值 PnL 更新时记录"""
        j = self._journals.get(trade_id)
        if j is None:
            return
        j.add(TradeEvent(
            ts         = time.time(),
            event_type = EVENT_PEAK_PNL_UPDATE,
            price      = price,
            peak_pnl   = peak_pnl,
            cur_ret    = cur_ret,
        ))

    def close(
        self,
        trade_id:     str,
        exit_price:   float,
        exit_reason:  str,
        lc_state:     str,
        aligned_score: float,
        velocity:     float,
        peak_pnl:     float,
        drawdown:     float,
        cur_ret:      float,
        pnl_usdt:     float,
        fee_usdt:     float,
        ret_pct:      float,
        ret_lev_pct:  float,
        hold_seconds: float,
    ):
        """
        平仓：
          1. 写入 exit 事件到轨迹
          2. 导出单笔轨迹 CSV（trades/<trade_id>.csv）
          3. 追加到全局 trades.csv 和 exit_log.csv
          4. 清理内存
        """
        j = self._journals.get(trade_id)
        if j is None:
            return

        now     = time.time()
        net_pnl = pnl_usdt - fee_usdt

        # ── 1. 写入 exit 事件 ─────────────────────────────────────────────────
        j.add(TradeEvent(
            ts            = now,
            event_type    = EVENT_EXIT,
            price         = exit_price,
            lc_state      = lc_state,
            aligned_score = aligned_score,
            velocity      = velocity,
            cur_ret       = cur_ret,
            peak_pnl      = peak_pnl,
            drawdown      = drawdown,
            exit_reason   = exit_reason,
        ))

        # ── 2. 导出单笔轨迹 CSV ───────────────────────────────────────────────
        trail_path = os.path.join(self._trail_dir, f"{trade_id}.csv")
        trail_cols = [
            "ts", "secs_from_entry", "event_type", "price",
            "lc_state", "aligned_score", "velocity",
            "cur_ret", "peak_pnl", "drawdown", "exit_reason", "extra",
        ]
        try:
            with open(trail_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=trail_cols)
                writer.writeheader()
                for ev in j.events:
                    writer.writerow({
                        "ts":              round(ev.ts, 3),
                        "secs_from_entry": round(ev.ts - j.entry_time, 1),
                        "event_type":      ev.event_type,
                        "price":           round(ev.price, 6),
                        "lc_state":        ev.lc_state,
                        "aligned_score":   round(ev.aligned_score, 4),
                        "velocity":        round(ev.velocity, 5),
                        "cur_ret":         round(ev.cur_ret, 5),
                        "peak_pnl":        round(ev.peak_pnl, 4),
                        "drawdown":        round(ev.drawdown, 5),
                        "exit_reason":     ev.exit_reason,
                        "extra":           ev.extra,
                    })
        except Exception as e:
            from data_layer.logger import logger
            logger.debug(f"[TradeRecorder] 轨迹 CSV 写入失败 {trade_id}: {e}")

        # ── 3a. 追加到 trades.csv ─────────────────────────────────────────────
        self._append_csv(self._trades_path, self._TRADES_COLS, {
            "trade_id":             trade_id,
            "symbol":               j.symbol,
            "side":                 j.side,
            "entry_time":           round(j.entry_time, 3),
            "exit_time":            round(now, 3),
            "entry_price":          round(j.entry_price, 6),
            "exit_price":           round(exit_price, 6),
            "qty":                  round(j.qty, 6),
            "leverage":             j.leverage,
            "open_vol":             round(j.open_vol, 5),
            "entry_score":          round(j.entry_score, 4),
            "exit_reason":          exit_reason,
            "hold_seconds":         round(hold_seconds, 1),
            "ret_pct":              round(ret_pct, 4),
            "ret_lev_pct":          round(ret_lev_pct, 4),
            "pnl_usdt":             round(pnl_usdt, 4),
            "fee_usdt":             round(fee_usdt, 4),
            "net_pnl":              round(net_pnl, 4),
            "lc_state_at_exit":     lc_state,
            "aligned_score_at_exit": round(aligned_score, 4),
            "velocity_at_exit":     round(velocity, 5),
            "peak_pnl":             round(peak_pnl, 4),
            "drawdown_at_exit":     round(drawdown, 5),
        })

        # ── 3b. 追加到 exit_log.csv ───────────────────────────────────────────
        self._append_csv(self._exit_path, self._EXIT_COLS, {
            "ts":            round(now, 3),
            "trade_id":      trade_id,
            "symbol":        j.symbol,
            "side":          j.side,
            "exit_reason":   exit_reason,
            "lc_state":      lc_state,
            "aligned_score": round(aligned_score, 4),
            "velocity":      round(velocity, 5),
            "cur_ret":       round(cur_ret, 5),
            "peak_pnl":      round(peak_pnl, 4),
            "drawdown":      round(drawdown, 5),
            "entry_price":   round(j.entry_price, 6),
            "exit_price":    round(exit_price, 6),
            "hold_seconds":  round(hold_seconds, 1),
            "pnl_usdt":      round(pnl_usdt, 4),
            "fee_usdt":      round(fee_usdt, 4),
            "net_pnl":       round(net_pnl, 4),
            "ret_pct":       round(ret_pct, 4),
            "ret_lev_pct":   round(ret_lev_pct, 4),
        })

        # ── 4. 清理内存 ───────────────────────────────────────────────────────
        self._active.pop((j.symbol, j.side), None)
        del self._journals[trade_id]

    # ─── 工具方法 ─────────────────────────────────────────────────────────────

    def _init_csv(self, path: str, cols: list):
        """文件不存在时写入表头"""
        if not os.path.exists(path):
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=cols).writeheader()
            except Exception:
                pass

    def _reset_csv(self, path: str, cols: list):
        """强制覆盖文件，写入新表头（清空历史数据）"""
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=cols).writeheader()
        except Exception:
            pass

    def _append_csv(self, path: str, cols: list, row: dict):
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                writer.writerow(row)
        except Exception as e:
            from data_layer.logger import logger
            logger.debug(f"[TradeRecorder] CSV 写入失败 {path}: {e}")

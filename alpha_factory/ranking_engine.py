"""
Ranking Engine - 全市场横截面排序（多空双向）

做多信号：横截面得分最高的 Top N（强势异动，还没涨）
做空信号：横截面得分最低的 Bottom N（成交量萎缩，OFI 负，已超涨）

信号稳定性：
    EMA 平滑    : 历史分数衰减加权，抑制单次爆量噪音
    confirm_rounds : 品种需连续进入候选池 N 次才确认进场

注：退出由策略层的止盈止损负责，RankingEngine 只负责开仓信号。
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
from data_layer.logger import logger


@dataclass
class RankEntry:
    symbol:        str
    score:         float   # EMA 平滑分数
    raw_score:     float   # 本轮瞬时分数
    rank:          int
    side:          str     # "LONG" 或 "SHORT"
    confirm_count: int = 0
    miss_count:    int = 0


class RankingEngine:
    """
    全市场多空双向排序器。

    rank() 返回：
        (long_confirmed, new_longs, exit_longs,
         short_confirmed, new_shorts, exit_shorts)
    """

    def __init__(
        self,
        top_n:                int   = 3,     # 最大同时做多品种数
        bottom_n:             int   = 3,     # 最大同时做空品种数
        long_score_threshold:  float = 0.5,  # 做多最低 EMA 分数
        short_score_threshold: float = -0.5, # 做空最高 EMA 分数（负值）
        ema_alpha:             float = 0.4,
        confirm_rounds:        int   = 2,
    ):
        self.top_n                 = top_n
        self.bottom_n              = bottom_n
        self.long_score_threshold  = long_score_threshold
        self.short_score_threshold = short_score_threshold
        self.ema_alpha             = ema_alpha
        self.confirm_rounds        = confirm_rounds

        self._ema_scores: Dict[str, float] = {}

        # 候选池（各自独立追踪）
        self._long_candidates:  Dict[str, RankEntry] = {}
        self._short_candidates: Dict[str, RankEntry] = {}

        # 已确认持仓
        self._confirmed_longs:  Dict[str, RankEntry] = {}
        self._confirmed_shorts: Dict[str, RankEntry] = {}

        self._rank_count = 0

    def rank(
        self, scores: Dict[str, float]
    ) -> Tuple[
        List[RankEntry], List[RankEntry], List[str],   # long
        List[RankEntry], List[RankEntry], List[str],   # short
    ]:
        if not scores:
            return [], [], [], [], [], []

        # ── 1. 更新 EMA ──────────────────────────────────────────────────────
        for sym, raw in scores.items():
            prev = self._ema_scores.get(sym, raw)
            self._ema_scores[sym] = self.ema_alpha * raw + (1 - self.ema_alpha) * prev

        # ── 2. 按 EMA 分数排序 ────────────────────────────────────────────────
        sorted_ema = sorted(self._ema_scores.items(), key=lambda x: -x[1])

        # 做多候选池：Top N*3，EMA > long_score_threshold
        long_pool = {
            sym for sym, ema in sorted_ema[: self.top_n * 3]
            if ema >= self.long_score_threshold
        }

        # 做空候选池：Bottom N*3，EMA < short_score_threshold
        short_pool = {
            sym for sym, ema in sorted_ema[-(self.bottom_n * 3):]
            if ema <= self.short_score_threshold
        }

        # 同一品种不能同时在多空候选池（极端情况保护）
        short_pool -= long_pool

        # ── 3. 更新候选计数 ───────────────────────────────────────────────────
        self._update_candidates(self._long_candidates,  long_pool,  scores, "LONG")
        self._update_candidates(self._short_candidates, short_pool, scores, "SHORT")

        # ── 4. 确认进场 ───────────────────────────────────────────────────────
        new_longs  = self._confirm_entries(self._long_candidates,  self._confirmed_longs,  self.top_n)
        new_shorts = self._confirm_entries(self._short_candidates, self._confirmed_shorts, self.bottom_n)

        # 同品种不允许同时持多空（已是多仓的不做空，反之亦然）
        for sym in list(self._confirmed_longs.keys()):
            if sym in self._confirmed_shorts:
                del self._confirmed_shorts[sym]
        new_shorts = [e for e in new_shorts if e.symbol not in self._confirmed_longs]

        # ── 5. 更新排名序号 ────────────────────────────────────────────────────
        self._assign_ranks(self._confirmed_longs,  reverse=True)   # 分数高 = 排名1
        self._assign_ranks(self._confirmed_shorts, reverse=False)   # 分数低 = 排名1

        self._rank_count += 1

        # ── 6. 日志 ────────────────────────────────────────────────────────────
        self._log(new_longs, new_shorts)

        return (
            list(self._confirmed_longs.values()),  new_longs,  [],
            list(self._confirmed_shorts.values()), new_shorts, [],
        )

    # ─── 内部辅助 ────────────────────────────────────────────────────────────

    def _update_candidates(
        self,
        candidates: Dict[str, RankEntry],
        pool: set,
        scores: Dict[str, float],
        side: str,
    ):
        """更新候选池的 confirm_count / miss_count"""
        # 已在候选池的品种
        for sym in list(candidates.keys()):
            entry = candidates[sym]
            raw   = scores.get(sym, 0.0)
            ema   = self._ema_scores.get(sym, 0.0)
            if sym in pool:
                entry.confirm_count += 1
                entry.miss_count     = 0
                entry.score          = ema
                entry.raw_score      = raw
            else:
                entry.miss_count    += 1
                entry.confirm_count  = 0

        # 新出现在候选池的品种
        for sym in pool - set(candidates.keys()):
            candidates[sym] = RankEntry(
                symbol        = sym,
                score         = self._ema_scores.get(sym, 0.0),
                raw_score     = scores.get(sym, 0.0),
                rank          = 0,
                side          = side,
                confirm_count = 1,
                miss_count    = 0,
            )

        # 清理长期消失的品种（连续3轮未出现则移除）
        stale = [s for s, e in candidates.items() if e.miss_count > 3]
        for sym in stale:
            del candidates[sym]

    def _confirm_entries(
        self,
        candidates: Dict[str, RankEntry],
        confirmed:  Dict[str, RankEntry],
        max_n: int,
    ) -> List[RankEntry]:
        """把满足 confirm_rounds 的品种加入已确认集合"""
        new_entries = []
        eligible = sorted(
            [e for e in candidates.values()
             if e.confirm_count >= self.confirm_rounds and e.symbol not in confirmed],
            key=lambda e: -e.score,
        )
        for entry in eligible:
            if len(confirmed) >= max_n:
                break
            confirmed[entry.symbol] = entry
            new_entries.append(entry)
        return new_entries

    def _assign_ranks(self, confirmed: Dict[str, RankEntry], reverse: bool):
        for i, entry in enumerate(
            sorted(confirmed.values(), key=lambda e: -e.score if reverse else e.score),
            start=1,
        ):
            entry.rank = i

    def _log(self, new_longs, new_shorts):
        long_str = " | ".join(
            f"#{e.rank} {e.symbol}(ema={e.score:+.3f})"
            for e in sorted(self._confirmed_longs.values(), key=lambda e: e.rank)
        ) or "空"
        short_str = " | ".join(
            f"#{e.rank} {e.symbol}(ema={e.score:+.3f})"
            for e in sorted(self._confirmed_shorts.values(), key=lambda e: e.rank)
        ) or "空"
        logger.info(
            f"[Ranking] #{self._rank_count}"
            f" | 多: [{long_str}]  新进={[e.symbol for e in new_longs]}"
            f" | 空: [{short_str}] 新进={[e.symbol for e in new_shorts]}"
        )

    # ─── 状态查询 ────────────────────────────────────────────────────────────

    @property
    def current_longs(self) -> List[str]:
        return [e.symbol for e in sorted(self._confirmed_longs.values(),  key=lambda e: e.rank)]

    @property
    def current_shorts(self) -> List[str]:
        return [e.symbol for e in sorted(self._confirmed_shorts.values(), key=lambda e: e.rank)]

    def release_long(self, symbol: str):
        """策略平多仓后调用，从已确认多头集合中移除，为新信号腾出位置。"""
        self._confirmed_longs.pop(symbol, None)
        self._long_candidates.pop(symbol, None)

    def release_short(self, symbol: str):
        """策略平空仓后调用，从已确认空头集合中移除，为新信号腾出位置。"""
        self._confirmed_shorts.pop(symbol, None)
        self._short_candidates.pop(symbol, None)

    def get_score(self, symbol: str) -> float:
        return self._ema_scores.get(symbol, 0.0)

    @property
    def rank_count(self) -> int:
        return self._rank_count

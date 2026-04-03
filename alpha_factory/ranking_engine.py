"""
Ranking Engine - 全市场横截面排序（多空双向，升级版）

升级点：
    1. 候选池不再固定 Top N * 3，而是“自适应强度筛选”
    2. 引入 long_margin / short_margin，过滤边缘信号
    3. confirm 不再只是“连续出现”，还要求分数不明显衰减
    4. 保底机制：候选过少时，至少保留 top_n / bottom_n 个

做多信号：
    横截面得分高，且高于动态阈值

做空信号：
    横截面得分低，且低于动态阈值

信号稳定性：
    EMA 平滑        : 历史分数衰减加权，抑制单次爆量噪音
    confirm_rounds  : 品种需连续进入候选池 N 次才确认进场
    decay filter    : 若候选分数明显衰减，则 confirm_count 清零

注：
    RankingEngine 只负责“筛选谁值得做”，
    真正进场由策略层 / timing 层决定。
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
    全市场多空双向排序器（升级版）。

    rank() 返回：
        (long_confirmed, new_longs, exit_longs,
         short_confirmed, new_shorts, exit_shorts)
    """

    def __init__(
        self,
        top_n:                 int   = 3,      # 最大同时做多品种数
        bottom_n:              int   = 3,      # 最大同时做空品种数
        long_score_threshold:  float = 0.5,    # 多头基础阈值
        short_score_threshold: float = -0.5,   # 空头基础阈值
        ema_alpha:             float = 0.4,
        confirm_rounds:        int   = 3,      # 升级：默认 3 轮确认
        long_margin:           float = 0.20,   # 升级：多头动态阈值 margin
        short_margin:          float = 0.20,   # 升级：空头动态阈值 margin
        pool_expand:           int   = 3,      # 只在 top_n * pool_expand 范围内估计动态阈值
        min_candidates:        int   = 0,      # 保底候选数；0 表示自动 = top_n / bottom_n
        decay_tolerance:       float = 0.95,   # 升级：候选分数衰减容忍度
        stale_rounds:          int   = 3,      # 连续几轮未出现后清理候选
    ):
        self.top_n                 = top_n
        self.bottom_n              = bottom_n
        self.long_score_threshold  = long_score_threshold
        self.short_score_threshold = short_score_threshold
        self.ema_alpha             = ema_alpha
        self.confirm_rounds        = confirm_rounds

        self.long_margin           = long_margin
        self.short_margin          = short_margin
        self.pool_expand           = max(1, pool_expand)
        self.min_candidates        = min_candidates
        self.decay_tolerance       = decay_tolerance
        self.stale_rounds          = stale_rounds

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
        if not sorted_ema:
            return [], [], [], [], [], []

        # ── 3. 动态阈值筛选候选池（升级核心）─────────────────────────────────
        long_pool, dynamic_long_thresh = self._build_long_pool(sorted_ema)
        short_pool, dynamic_short_thresh = self._build_short_pool(sorted_ema)

        # 同一品种不能同时在多空候选池（极端情况保护）
        short_pool -= long_pool

        # ── 4. 更新候选计数（加入“分数衰减过滤”）──────────────────────────────
        self._update_candidates(self._long_candidates,  long_pool,  scores, "LONG")
        self._update_candidates(self._short_candidates, short_pool, scores, "SHORT")

        # ── 5. 确认进场 ───────────────────────────────────────────────────────
        new_longs  = self._confirm_entries(self._long_candidates,  self._confirmed_longs,  self.top_n, side="LONG")
        new_shorts = self._confirm_entries(self._short_candidates, self._confirmed_shorts, self.bottom_n, side="SHORT")

        # 同品种不允许同时持多空（已是多仓的不做空，反之亦然）
        for sym in list(self._confirmed_longs.keys()):
            if sym in self._confirmed_shorts:
                del self._confirmed_shorts[sym]
        new_shorts = [e for e in new_shorts if e.symbol not in self._confirmed_longs]

        # ── 6. 更新排名序号 ────────────────────────────────────────────────────
        self._assign_ranks(self._confirmed_longs,  reverse=True)    # 分数高 = 排名1
        self._assign_ranks(self._confirmed_shorts, reverse=False)   # 分数低 = 排名1

        self._rank_count += 1

        # ── 7. 日志 ───────────────────────────────────────────────────────────
        self._log(new_longs, new_shorts, dynamic_long_thresh, dynamic_short_thresh, len(long_pool), len(short_pool))

        return (
            list(self._confirmed_longs.values()),  new_longs,  [],
            list(self._confirmed_shorts.values()), new_shorts, [],
        )

    # ─── 候选池构建 ──────────────────────────────────────────────────────────

    def _build_long_pool(self, sorted_ema: List[Tuple[str, float]]) -> Tuple[set, float]:
        """
        多头候选池：
            1. 只看前 top_n * pool_expand 个强势区间
            2. 动态阈值 = max(基础阈值, top_n均值 - long_margin)
            3. 保底：至少保留 min_candidates 个
        """
        top_k = max(self.top_n * self.pool_expand, self.top_n)
        top_slice = sorted_ema[:top_k]

        top_scores = [ema for _, ema in top_slice]
        top_core_scores = [ema for _, ema in sorted_ema[:self.top_n]]

        if top_core_scores:
            core_avg = sum(top_core_scores) / len(top_core_scores)
            dynamic_long_thresh = max(self.long_score_threshold, core_avg - self.long_margin)
        else:
            dynamic_long_thresh = self.long_score_threshold

        long_pool = {
            sym for sym, ema in top_slice
            if ema >= dynamic_long_thresh
        }

        min_keep = self.min_candidates if self.min_candidates > 0 else self.top_n
        if len(long_pool) < min_keep:
            long_pool = {sym for sym, _ in sorted_ema[:min_keep]}

        return long_pool, dynamic_long_thresh

    def _build_short_pool(self, sorted_ema: List[Tuple[str, float]]) -> Tuple[set, float]:
        """
        空头候选池：
            1. 只看后 bottom_n * pool_expand 个弱势区间
            2. 动态阈值 = min(基础阈值, bottom_n均值 + short_margin)
            3. 保底：至少保留 min_candidates 个
        """
        bottom_k = max(self.bottom_n * self.pool_expand, self.bottom_n)
        bottom_slice = sorted_ema[-bottom_k:]

        # 注意：sorted_ema 是从大到小，底部的前 bottom_n 个应该从最弱端取
        bottom_core_scores = [ema for _, ema in sorted_ema[-self.bottom_n:]]

        if bottom_core_scores:
            core_avg = sum(bottom_core_scores) / len(bottom_core_scores)
            dynamic_short_thresh = min(self.short_score_threshold, core_avg + self.short_margin)
        else:
            dynamic_short_thresh = self.short_score_threshold

        short_pool = {
            sym for sym, ema in bottom_slice
            if ema <= dynamic_short_thresh
        }

        min_keep = self.min_candidates if self.min_candidates > 0 else self.bottom_n
        if len(short_pool) < min_keep:
            short_pool = {sym for sym, _ in sorted_ema[-min_keep:]}

        return short_pool, dynamic_short_thresh

    # ─── 内部辅助 ────────────────────────────────────────────────────────────

    def _update_candidates(
        self,
        candidates: Dict[str, RankEntry],
        pool: set,
        scores: Dict[str, float],
        side: str,
    ):
        """
        更新候选池的 confirm_count / miss_count

        升级点：
            - 若仍在候选池，但分数明显衰减，则 confirm_count 清零
            - 若稳定/增强，则 confirm_count += 1
        """
        # 已在候选池的品种
        for sym in list(candidates.keys()):
            entry = candidates[sym]
            raw   = scores.get(sym, 0.0)
            ema   = self._ema_scores.get(sym, 0.0)

            if sym in pool:
                prev_score = entry.score
                curr_score = ema

                # 分数不能显著衰减；允许轻微回落，但不能“候选还在，强度已经掉了”
                if prev_score == 0.0 or curr_score >= prev_score * self.decay_tolerance:
                    entry.confirm_count += 1
                else:
                    entry.confirm_count = 0

                entry.miss_count = 0
                entry.score = curr_score
                entry.raw_score = raw
            else:
                entry.miss_count += 1
                entry.confirm_count = 0

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

        # 清理长期消失的品种
        stale = [s for s, e in candidates.items() if e.miss_count > self.stale_rounds]
        for sym in stale:
            del candidates[sym]

    def _confirm_entries(
        self,
        candidates: Dict[str, RankEntry],
        confirmed: Dict[str, RankEntry],
        max_n: int,
        side: str,
    ) -> List[RankEntry]:
        """
        把满足 confirm_rounds 的品种加入已确认集合。
        LONG 按分数高到低选，SHORT 按分数低到高选。
        """
        new_entries = []

        eligible = [
            e for e in candidates.values()
            if e.confirm_count >= self.confirm_rounds and e.symbol not in confirmed
        ]

        if side == "LONG":
            eligible = sorted(eligible, key=lambda e: -e.score)
        else:
            eligible = sorted(eligible, key=lambda e: e.score)

        for entry in eligible:
            if len(confirmed) >= max_n:
                break
            confirmed[entry.symbol] = entry
            new_entries.append(entry)

        return new_entries

    def _assign_ranks(self, confirmed: Dict[str, RankEntry], reverse: bool):
        ordered = sorted(
            confirmed.values(),
            key=lambda e: -e.score if reverse else e.score
        )
        for i, entry in enumerate(ordered, start=1):
            entry.rank = i

    def _log(self, new_longs, new_shorts, long_thresh, short_thresh, long_pool_size, short_pool_size):
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
            f" | 多阈值={long_thresh:+.3f} 候选={long_pool_size}"
            f" | 多: [{long_str}] 新进={[e.symbol for e in new_longs]}"
            f" | 空阈值={short_thresh:+.3f} 候选={short_pool_size}"
            f" | 空: [{short_str}] 新进={[e.symbol for e in new_shorts]}"
        )

    # ─── 状态查询 ────────────────────────────────────────────────────────────

    @property
    def current_longs(self) -> List[str]:
        return [e.symbol for e in sorted(self._confirmed_longs.values(), key=lambda e: e.rank)]

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
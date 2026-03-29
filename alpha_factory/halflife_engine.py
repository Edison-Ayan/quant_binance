"""
HalfLifeEngine - Alpha 半衰期在线估计引擎

核心思想：
    alpha 不是永恒的，它有生命周期：
        E[r(t+τ) | signal(t)] ≈ IC₀ · exp(-λτ)

    半衰期 t_half = ln(2) / λ 告诉你：
    - 信号在 t_half 秒后预测力下降到初始值的 50%
    - 持仓不应显著超过 2 × t_half

工作原理：
    1. 每轮 _rank_and_trade 记录一个快照：
       {ts, symbol → (composite_score, lob_score, flow_score, price)}

    2. 每轮用「N轮前的快照」计算实际收益，与当时的信号做 IC:
       IC(τ) = corr(score_at_t, return_{t→t+τ})

    3. 对 IC(τ) curve 做指数衰减拟合：
       ln(IC(τ)) = ln(IC₀) - λτ  →  OLS 求 λ

    4. 输出：
       - get_halflife(group)          → 该因子组的半衰期（秒）
       - effective_score(score, hold) → score × exp(-hold/t_half)
       - get_ic_curve()               → {τ: IC} 字典，供 Dashboard 展示

因子组划分：
    composite : 全部因子加权后的综合得分
    lob       : lob_z1 + lob_z2 + lob_z3 的简单均值（LOB 流形因子）
    flow      : ofi + oi_change_pct 的简单均值（订单流因子）
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from data_layer.logger import logger


# ── 配置 ──────────────────────────────────────────────────────────────────────

# 检验多少轮之前的信号（每轮 = rank_interval 秒）
# [1, 2, 4, 8] 轮 × 60s = [60s, 120s, 240s, 480s]
HORIZON_ROUNDS: List[int] = [1, 2, 4, 8]

# 每个时间点 IC 的 EMA 衰减系数（用于平滑噪声）
IC_EMA_ALPHA: float = 0.25

# 曲线拟合所需最少非零 IC 数据点
MIN_POINTS_FOR_FIT: int = 2

# 半衰期合法范围（秒）：低于 15s 认为数据噪声太大，高于 1800s 退化为静态
HALFLIFE_MIN_S: float = 15.0
HALFLIFE_MAX_S: float = 1800.0

# 历史快照保留轮数
SNAPSHOT_BUFFER: int = 20

# 每组的默认半衰期（数据不足时使用）
DEFAULT_HALFLIFE: Dict[str, float] = {
    "composite": 180.0,   # 3分钟
    "lob":       60.0,    # 1分钟（LOB alpha 消失最快）
    "flow":      120.0,   # 2分钟（OFI 等流动性信号较持久）
}


class HalfLifeEngine:
    """
    在线 Alpha 半衰期估计。

    使用方式（在 alpha_strategy.py 中）：

        # __init__
        self.halflife_engine = HalfLifeEngine(rank_interval=60)

        # _rank_and_trade 中，打完分后立刻调用：
        self.halflife_engine.record_snapshot(scores, features, now)

        # 同一函数，在记录之前先更新 IC（用上一批快照 vs 当前价格）
        self.halflife_engine.update(features, now)

        # 在 _signal_exit 中使用：
        hl = self.halflife_engine.get_halflife("composite")
        eff_score = self.halflife_engine.effective_score(score, hold_time)
    """

    def __init__(self, rank_interval: int = 60):
        self._interval = rank_interval   # 每轮排序的秒数

        # 快照环形缓冲区：每轮一个条目
        # 每条：{"ts": float, "scores": {sym: float},
        #        "lob": {sym: float}, "flow": {sym: float}, "prices": {sym: float}}
        self._snapshots: deque = deque(maxlen=SNAPSHOT_BUFFER)

        # IC(τ) 的 EMA 平滑值，按因子组和轮数索引
        # {group: {n_rounds: ema_ic}}
        self._ic_ema: Dict[str, Dict[int, float]] = {
            group: {r: 0.0 for r in HORIZON_ROUNDS}
            for group in DEFAULT_HALFLIFE
        }

        # 当前半衰期估计（EMA 平滑）
        self._halflife: Dict[str, float] = dict(DEFAULT_HALFLIFE)

        # IC curve 供外部展示
        self._ic_curve: Dict[str, Dict[int, float]] = {
            group: {} for group in DEFAULT_HALFLIFE
        }

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def record_snapshot(
        self,
        scores:   Dict[str, float],
        features: dict,
        ts:       float,
    ):
        """
        记录本轮打分快照。

        在 _rank_and_trade 中打完分、做完信号退出之后调用。
        features 是 FeatureEngine.get_all_features() 的输出。
        """
        lob_scores:  Dict[str, float] = {}
        flow_scores: Dict[str, float] = {}
        prices:      Dict[str, float] = {}

        for sym, feat in features.items():
            # LOB 组：三个 PC 的均值
            lob_scores[sym]  = (feat.lob_z1 + feat.lob_z2 + feat.lob_z3) / 3.0
            # Flow 组：OFI 和 OI 变化的均值
            flow_scores[sym] = (feat.ofi + feat.oi_change_pct) / 2.0
            prices[sym]      = feat.last_price

        self._snapshots.append({
            "ts":     ts,
            "scores": dict(scores),
            "lob":    lob_scores,
            "flow":   flow_scores,
            "prices": prices,
        })

    def update(self, features: dict, now: float):
        """
        用「N轮前的快照」计算实际收益，更新各 IC(τ)，并重新拟合半衰期。

        在 _rank_and_trade 开头（record_snapshot 之前）调用，
        确保用的是上一轮及更早的快照。
        """
        snaps = list(self._snapshots)
        n_snaps = len(snaps)

        current_prices = {sym: feat.last_price for sym, feat in features.items()}

        for r in HORIZON_ROUNDS:
            idx = n_snaps - r - 1   # r 轮前的快照索引
            if idx < 0:
                continue

            past = snaps[idx]
            self._update_ic_for_horizon(past, current_prices, r)

        # 重新拟合半衰期
        self._fit_halflives()

    def get_halflife(self, group: str = "composite") -> float:
        """返回指定因子组的当前半衰期估计（秒）。"""
        return self._halflife.get(group, DEFAULT_HALFLIFE.get(group, 120.0))

    def effective_score(
        self,
        raw_score:    float,
        holding_time: float,
        group:        str = "composite",
    ) -> float:
        """
        按半衰期衰减调整后的有效得分。

        effective_score = raw_score × exp(-holding_time / t_half)

        用法：在 _signal_exit 中替换原来 alpha_half_life 参数的计算。
        """
        t_half = self.get_halflife(group)
        if t_half <= 0:
            return raw_score
        return raw_score * math.exp(-holding_time / t_half)

    def dynamic_time_stop(self, multiplier: float = 2.5) -> float:
        """
        根据当前半衰期动态建议的时间止损阈值（秒）。

        默认 = 2.5 × t_half（持仓超过这个时间，信号通常已衰减到初始值的 8% 以下）。
        """
        return max(60.0, self.get_halflife() * multiplier)

    def get_ic_curve(self, group: str = "composite") -> Dict[int, float]:
        """返回指定组的 IC(τ) 字典，键为秒数，值为 IC（供 Dashboard 展示）。"""
        return dict(self._ic_curve.get(group, {}))

    def get_status(self) -> dict:
        """返回引擎状态摘要（供日志和 Dashboard 使用）。"""
        return {
            "halflife_composite_s": round(self._halflife.get("composite", 0), 1),
            "halflife_lob_s":       round(self._halflife.get("lob",       0), 1),
            "halflife_flow_s":      round(self._halflife.get("flow",      0), 1),
            "snapshots":            len(self._snapshots),
            "ic_curve":             {g: self._ic_curve[g] for g in self._ic_curve},
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _update_ic_for_horizon(
        self,
        past:           dict,
        current_prices: Dict[str, float],
        n_rounds:       int,
    ):
        """
        计算 n_rounds 轮前快照的信号与当前收益的 IC，并 EMA 更新。
        """
        syms = [
            s for s in past["scores"]
            if s in current_prices
            and past["prices"].get(s, 0) > 0
            and current_prices[s] > 0
        ]
        if len(syms) < 10:
            return

        # 实际收益
        rets = [
            (current_prices[s] - past["prices"][s]) / past["prices"][s]
            for s in syms
        ]

        τ_s = n_rounds * self._interval   # 时间跨度（秒）

        for group in ("composite", "lob", "flow"):
            key = "scores" if group == "composite" else group
            signals = [past[key].get(s, 0.0) for s in syms]

            ic = self._pearson(signals, rets)
            if ic is None:
                continue

            old = self._ic_ema[group][n_rounds]
            new_ema = IC_EMA_ALPHA * ic + (1.0 - IC_EMA_ALPHA) * old
            self._ic_ema[group][n_rounds] = new_ema
            self._ic_curve[group][τ_s] = round(new_ema, 6)

    def _fit_halflives(self):
        """
        对每个因子组，用 OLS 对 ln(IC(τ)) ~ τ 做线性回归，推断 λ 和 t_half。

        模型：IC(τ) = IC₀ · exp(-λτ)
        → ln(IC(τ)) = ln(IC₀) - λτ
        → slope = -λ  (OLS)
        → t_half = ln(2) / λ
        """
        for group in ("composite", "lob", "flow"):
            curve = self._ic_curve.get(group, {})

            # 只用 IC > 0 的点（取对数要求正值）
            pts: List[Tuple[float, float]] = [
                (τ, math.log(ic))
                for τ, ic in sorted(curve.items())
                if ic > 0.005          # 过低的 IC 点噪声大，过滤
            ]

            if len(pts) < MIN_POINTS_FOR_FIT:
                continue

            taus    = [p[0] for p in pts]
            log_ics = [p[1] for p in pts]

            # OLS 估计 slope = -λ
            n    = len(pts)
            mean_t = sum(taus) / n
            mean_l = sum(log_ics) / n
            num  = sum((t - mean_t) * (l - mean_l) for t, l in zip(taus, log_ics))
            den  = sum((t - mean_t) ** 2 for t in taus)
            if den < 1e-10:
                continue

            slope = num / den      # 应为负值
            lam   = -slope         # λ > 0

            if lam <= 0:
                continue

            raw_hl = math.log(2.0) / lam
            raw_hl = max(HALFLIFE_MIN_S, min(HALFLIFE_MAX_S, raw_hl))

            # EMA 平滑半衰期（避免单轮跳变）
            old_hl = self._halflife.get(group, DEFAULT_HALFLIFE[group])
            new_hl = 0.2 * raw_hl + 0.8 * old_hl
            self._halflife[group] = new_hl

        logger.debug(
            f"[HalfLife] t_half composite={self._halflife['composite']:.0f}s "
            f"lob={self._halflife['lob']:.0f}s "
            f"flow={self._halflife['flow']:.0f}s"
        )

    @staticmethod
    def _pearson(x: List[float], y: List[float]) -> Optional[float]:
        """Pearson 相关系数，样本不足或标准差为零时返回 None。"""
        n = len(x)
        if n < 5:
            return None
        mx = sum(x) / n
        my = sum(y) / n
        cov  = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
        sx   = (sum((a - mx) ** 2 for a in x) / n) ** 0.5
        sy   = (sum((b - my) ** 2 for b in y) / n) ** 0.5
        if sx < 1e-10 or sy < 1e-10:
            return None
        return cov / (sx * sy)

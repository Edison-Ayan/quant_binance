"""
Scoring Engine - 横截面慢因子打分 → 候选池（中频版定稿）

职责：
    接收所有品种的特征向量，通过横截面标准化 + 加权求和，
    为每个品种计算一个综合得分，供 RankingEngine 选出候选池。

定位（双层架构中的慢层）：
    只回答“选谁”（候选池 Top N / Bottom N），
    不回答“什么时候进”（实时进场时机由 LOBTimingEngine 负责）。

设计原则（中频版）：
    1. 慢层要“稳定选股”，不能过于追逐短噪声
    2. 资金流 / 中短趋势 / 24h 背景趋势共同决定候选优先级
    3. funding_rate 作为拥挤修正因子，方向为反向
    4. MEAN_REVERTING 市场下自动翻转动量类因子方向
    5. 同时保留 raw score 与 normalized score：
       - normalized score 供 RankingEngine 排序
       - raw score 可供策略层做强度 gating / 调试解释
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .feature_engine import SymbolFeatures
from data_layer.logger import logger


# ── 因子权重（中频慢层版）────────────────────────────────────────────────────
# 说明：
# - ret_24h 作为慢层锚，权重最高
# - ret_5m 作为中期趋势确认
# - volume_zscore / oi_change_pct 作为资金流与参与度确认
# - ret_1m 只保留少量权重，避免过快
# - funding_rate 作为拥挤修正，小权重反向
FACTOR_WEIGHTS: Dict[str, float] = {
    "volume_zscore":  0.25,
    "oi_change_pct":  0.25,
    "ret_1m":         0.05,
    "ret_5m":         0.05,
    "funding_rate":   0.35,
    "ret_24h":        0.05,
}

# ── 因子方向修正符 ─────────────────────────────────────────────────────────────
# +1.0 = 正向（高值 → 更做多）
# -1.0 = 反向（高值 → 更做空）
FACTOR_SIGNS: Dict[str, float] = {
    "volume_zscore":  +1.0,   # 净买压显著高于历史常态 → 正信号
    "oi_change_pct":  +1.0,   # OI 增加 + 价格方向一致 → 趋势强化
    "ret_1m":         +1.0,   # 短周期动量，仅作轻辅助
    "ret_5m":         +1.0,   # 中期趋势确认
    "funding_rate":   +1.0,   # 资金费率高 → 多头拥挤 → 正向修正
    "ret_24h":        +1.0,   # 24h 背景趋势
}

# 均值回归市场下自动翻转方向的因子（追涨→逆势）
MEAN_REVERTING_FLIP_FACTORS = {"ret_1m", "ret_5m", "ret_24h"}

# 横截面最少品种数（少于此数不计算打分）
MIN_SYMBOLS = 10

# 异常值截断（防止极端值拉偏横截面分布）
WINSORIZE_SIGMA = 3.0

# 是否在 compute_scores 时打印 top5
DEBUG_TOP5 = True

# 各因子取值 getters（统一管理，避免散落）
_FACTOR_GETTERS = {
    "volume_zscore": lambda feat: feat.volume_zscore,
    "oi_change_pct": lambda feat: feat.oi_change_pct,
    "ret_1m":        lambda feat: feat.ret_1m,
    "ret_5m":        lambda feat: feat.ret_5m,
    "funding_rate":  lambda feat: feat.funding_rate,
    "ret_24h":       lambda feat: feat.ret_24h,
}


class ScoringEngine:
    """
    横截面慢因子打分引擎。

    每个 rank_interval 调用一次 compute_scores()，
    返回 symbol → normalized_score 的字典，值越大代表做多信号越强。

    同时内部缓存最近一次：
        - raw composite score
        - normalized composite score
        - 各因子的横截面 z-score

    权重支持动态进化：update_weights(ic_dict)
    但实盘中频阶段建议先用固定权重跑稳定，再谨慎启用。
    """

    def __init__(self):
        # 运行时权重（可被 update_weights 动态调整，初始值 = FACTOR_WEIGHTS）
        self._weights: Dict[str, float] = dict(FACTOR_WEIGHTS)

        # 当前市场制度（"TRENDING" / "MEAN_REVERTING" / "VOLATILE" / "QUIET"）
        self._regime: str = "TRENDING"

        # 最近一次打分缓存（供调试 / breakdown / 策略层读取）
        self._last_raw_scores: Dict[str, float] = {}
        self._last_norm_scores: Dict[str, float] = {}
        self._last_factor_zscores: Dict[str, Dict[str, float]] = {}

    # ─── 市场制度 ─────────────────────────────────────────────────────────────

    def set_regime(self, regime: str):
        """
        由 AlphaFactory 在每次 rank 前调用，传入 MarketStateEngine 的 regime。

        MEAN_REVERTING 制度下：
            ret_1m / ret_5m / ret_24h 自动取反，避免追涨杀跌。
        """
        if regime != self._regime:
            logger.info(
                f"[ScoringEngine] 市场制度切换: {self._regime} → {regime}，"
                f"动量因子方向{'翻转' if regime == 'MEAN_REVERTING' else '恢复'}"
            )
        self._regime = regime

    # ─── 主打分逻辑 ───────────────────────────────────────────────────────────

    def compute_scores(self, features: Dict[str, SymbolFeatures]) -> Dict[str, float]:
        """
        对全市场特征向量进行横截面打分。

        步骤：
        1. 提取各因子的原始值矩阵
        2. 横截面 Z-score 归一化
        3. winsorize 截断
        4. 方向修正（含 regime flip）
        5. 按权重加权求和，得到 raw composite
        6. 对 raw composite 再做横截面归一化，得到 normalized composite

        返回：
            Dict[str, float] : {symbol: normalized_score}
            空字典表示样本不足，不应触发交易
        """
        if len(features) < MIN_SYMBOLS:
            logger.debug(f"[ScoringEngine] 样本不足 ({len(features)} < {MIN_SYMBOLS})，跳过打分")
            self._last_raw_scores = {}
            self._last_norm_scores = {}
            self._last_factor_zscores = {}
            return {}

        symbols = list(features.keys())
        n = len(symbols)

        # ── 提取原始因子值（跳过权重为 0 的因子）───────────────────────────────
        raw_factors: Dict[str, np.ndarray] = {}
        for name, getter in _FACTOR_GETTERS.items():
            if self._weights.get(name, 0.0) == 0.0:
                continue
            raw_factors[name] = np.array([getter(features[s]) for s in symbols], dtype=float)

        if not raw_factors:
            self._last_raw_scores = {}
            self._last_norm_scores = {}
            self._last_factor_zscores = {}
            return {}

        # ── 横截面 z-score + 方向修正 + 加权求和 ───────────────────────────────
        composite = np.zeros(n, dtype=float)

        # 记录每个 symbol 的因子 z-score（已含 direction / regime 修正）
        factor_zscores_by_symbol: Dict[str, Dict[str, float]] = {sym: {} for sym in symbols}

        for factor_name, arr in raw_factors.items():
            mu = arr.mean()
            sigma = arr.std()

            if sigma < 1e-8:
                # 所有品种该因子值几乎相同，无区分度，跳过
                continue

            z = (arr - mu) / sigma
            z = np.clip(z, -WINSORIZE_SIGMA, WINSORIZE_SIGMA)

            sign = FACTOR_SIGNS.get(factor_name, 1.0)

            # 均值回归制度下翻转动量因子方向
            if self._regime == "MEAN_REVERTING" and factor_name in MEAN_REVERTING_FLIP_FACTORS:
                sign = -sign

            z_signed = sign * z
            weight = self._weights.get(factor_name, 0.0)

            composite += weight * z_signed

            for i, sym in enumerate(symbols):
                factor_zscores_by_symbol[sym][factor_name] = float(z_signed[i])

        # ── 保存 raw composite（很重要，供后续强度过滤使用）───────────────────
        raw_scores = dict(zip(symbols, composite.tolist()))

        # ── normalized composite（供 RankingEngine 排序）─────────────────────
        c_std = composite.std()
        if c_std > 1e-8:
            composite_norm = (composite - composite.mean()) / c_std
        else:
            composite_norm = np.zeros(n, dtype=float)

        norm_scores = dict(zip(symbols, composite_norm.tolist()))

        self._last_raw_scores = raw_scores
        self._last_norm_scores = norm_scores
        self._last_factor_zscores = factor_zscores_by_symbol

        if DEBUG_TOP5:
            top5 = sorted(norm_scores.items(), key=lambda x: -x[1])[:5]
            top5_str = " | ".join(f"{s}:{v:.3f}" for s, v in top5)
            logger.debug(f"[ScoringEngine] Top5: {top5_str}")

        return norm_scores

    # ─── 结果读取 ─────────────────────────────────────────────────────────────

    def get_raw_score(self, symbol: str) -> float:
        """读取最近一次 compute_scores() 产生的 raw composite score。"""
        return self._last_raw_scores.get(symbol, 0.0)

    def get_normalized_score(self, symbol: str) -> float:
        """读取最近一次 compute_scores() 产生的 normalized score。"""
        return self._last_norm_scores.get(symbol, 0.0)

    def get_latest_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        返回最近一次打分结果：
            (raw_scores, normalized_scores)
        """
        return dict(self._last_raw_scores), dict(self._last_norm_scores)

    # ─── 因子分解 ─────────────────────────────────────────────────────────────

    def get_factor_breakdown(
        self,
        symbol: str,
        features: Dict[str, SymbolFeatures],
        use_cached: bool = True,
    ) -> Optional[Dict[str, float]]:
        """
        返回单个品种的因子分解（用于调试和解释）。

        返回内容包含：
            - 每个因子的 signed z-score
            - 每个因子的加权贡献
            - raw_total
            - normalized_total

        注意：
            raw_total 是更适合做“强度判断”的值，
            normalized_total 更适合用于排序比较。
        """
        if symbol not in features or len(features) < MIN_SYMBOLS:
            return None

        # 若当前缓存中没有结果，或显式要求重算，则触发一次 compute_scores
        if (not use_cached) or (symbol not in self._last_norm_scores):
            self.compute_scores(features)

        if symbol not in self._last_norm_scores:
            return None

        breakdown = {}

        zscores = self._last_factor_zscores.get(symbol, {})
        for factor_name, z_signed in zscores.items():
            breakdown[factor_name] = z_signed
            breakdown[f"{factor_name}_contrib"] = z_signed * self._weights.get(factor_name, 0.0)

        breakdown["raw_total"] = self._last_raw_scores.get(symbol, 0.0)
        breakdown["normalized_total"] = self._last_norm_scores.get(symbol, 0.0)

        return breakdown

    # ─── 权重更新（建议谨慎启用）──────────────────────────────────────────────

    def update_weights(self, ic_updates: Dict[str, float], alpha: float = 0.2):
        """
        基于近期因子 IC（预测力）动态调整权重（EMA 风格）。

        规则：
            有效 IC = 原始 IC × FACTOR_SIGNS（使反向因子的进化方向正确）
            IC > 0 → 预测力正 → 增权
            IC < 0 → 预测力负 → 减权
            权重约束在 [base × 0.3, base × 2.0] 之间
            更新后对全部权重做比例归一化，保持总权重不变

        注意：
            这套机制适合“样本较充分 + regime 较稳定”的阶段。
            在当前中频实盘早期，建议先观察，不要频繁启用。
        """
        changed = []

        for factor, ic in ic_updates.items():
            if factor not in self._weights:
                continue

            base = FACTOR_WEIGHTS.get(factor, 0.0)
            if base == 0.0:
                continue

            old_w = self._weights[factor]
            sign = FACTOR_SIGNS.get(factor, 1.0)
            effective_ic = ic * sign

            # 限制单次 ic 冲击，避免权重大幅乱跳
            clipped_ic = max(-1.0, min(1.0, effective_ic))
            adjustment = 1.0 + alpha * clipped_ic

            new_w = max(base * 0.3, min(base * 2.0, old_w * adjustment))
            self._weights[factor] = new_w

            if abs(new_w - old_w) > 1e-6:
                changed.append(f"{factor}:{old_w:.3f}→{new_w:.3f}")

        active = {k: v for k, v in self._weights.items() if v > 0}
        total = sum(active.values())

        if total > 0 and changed:
            target = sum(w for w in FACTOR_WEIGHTS.values() if w > 0)
            scale = target / total
            for f in active:
                self._weights[f] *= scale

            logger.info(f"[ScoringEngine] 权重进化: {' | '.join(changed)}")

    # ─── 运行时状态读取 ───────────────────────────────────────────────────────

    def get_current_weights(self) -> Dict[str, float]:
        """返回当前运行时权重（供报告展示）"""
        return dict(self._weights)

    def get_regime(self) -> str:
        """返回当前市场制度"""
        return self._regime
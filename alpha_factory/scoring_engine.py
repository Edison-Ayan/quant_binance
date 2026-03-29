"""
Scoring Engine - 横截面慢因子打分 → 候选池

职责：
    接收所有品种的特征向量，通过横截面标准化 + 加权求和，
    为每个品种计算一个综合得分，供 RankingEngine 选出候选池。

定位（双层架构中的慢层）：
    只回答"选谁"（候选池 Top N / Bottom N），
    不回答"什么时候进"（实时进场时机由 LOBTimingEngine 负责）。

因子权重设计（纯慢因子，60s 更新一次）：

    因子                权重    方向    含义
    ─────────────────────────────────────────────────────────
    volume_zscore       +0.30   +1     买卖方向加权成交量 Z-score
    oi_change_pct       +0.25   -1     OI 爆量 → 即将爆仓反转 → 反向
    ret_1m              +0.20   +1     1分钟动量（短期趋势延续）
    ret_5m              +0.15   +1     5分钟动量（中期趋势确认）
    funding_rate        +0.10   -1     资金费率极端高 → 多头拥挤 → 反向

标准化方式：
    - 横截面 Z-score 归一化（均值=0，标准差=1）
    - 异常值截断：±WINSORIZE_SIGMA σ
    - 不做 PCA 去市场因子（避免过滤掉真实的普涨/普跌信号）
"""

import numpy as np
from typing import Dict, Optional
from .feature_engine import SymbolFeatures
from data_layer.logger import logger


# ── 因子权重 ──────────────────────────────────────────────────────────────────
FACTOR_WEIGHTS: Dict[str, float] = {
    "volume_zscore":  0.30,
    "oi_change_pct":  0.25,
    "ret_1m":         0.20,
    "ret_5m":         0.15,
    "funding_rate":   0.10,
}

# ── 因子方向修正符 ─────────────────────────────────────────────────────────────
# +1.0 = 正向（高值 → 更做多）
# -1.0 = 反向（高值 → 更做空）
FACTOR_SIGNS: Dict[str, float] = {
    "volume_zscore":  +1.0,   # 主动买放量=正信号
    "oi_change_pct":  +1.0,   # OI 增加 = 新资金入场 → 正向
    "ret_1m":         +1.0,   # 短期动量延续
    "ret_5m":         +1.0,   # 中期趋势确认
    "funding_rate":   -1.0,   # 资金费率高 = 多头拥挤 → 反向
}

# 横截面最少品种数（少于此数不计算打分）
MIN_SYMBOLS = 10

# 异常值截断（防止极端值拉偏横截面分布）
WINSORIZE_SIGMA = 3.0

# 各因子取值 lambdas（统一管理，避免散落在多处）
_FACTOR_GETTERS = {
    "volume_zscore": lambda feat: feat.volume_zscore,
    "oi_change_pct": lambda feat: feat.oi_change_pct,
    "ret_1m":        lambda feat: feat.ret_1m,
    "ret_5m":        lambda feat: feat.ret_5m,
    "funding_rate":  lambda feat: feat.funding_rate,
}


class ScoringEngine:
    """
    横截面慢因子打分引擎。

    每个 rank_interval 调用一次 compute_scores()，
    返回 symbol → score 的字典，值越大代表做多信号越强。

    权重支持动态进化：update_weights(ic_dict) 根据近期 IC 自动调整。
    """

    def __init__(self):
        # 运行时权重（可被 update_weights 动态调整，初始值 = FACTOR_WEIGHTS）
        self._weights: Dict[str, float] = dict(FACTOR_WEIGHTS)

    def compute_scores(self, features: Dict[str, SymbolFeatures]) -> Dict[str, float]:
        """
        对全市场特征向量进行横截面打分。

        步骤：
        1. 提取各因子的原始值矩阵
        2. 截断异常值（±WINSORIZE_SIGMA σ）
        3. 横截面 Z-score 归一化 + 方向修正
        4. 按权重加权求和
        5. 最终横截面归一化（使分数尺度可解释）

        返回：
            Dict[str, float] : {symbol: composite_score}
            空字典表示样本不足，不应触发交易
        """
        if len(features) < MIN_SYMBOLS:
            logger.debug(f"[ScoringEngine] 样本不足 ({len(features)} < {MIN_SYMBOLS})，跳过打分")
            return {}

        symbols = list(features.keys())
        n = len(symbols)

        # ── 提取原始因子值（跳过权重为 0 的因子）────────────────────────────────
        raw_factors: Dict[str, np.ndarray] = {}
        for name, getter in _FACTOR_GETTERS.items():
            if self._weights.get(name, 0.0) == 0.0:
                continue
            raw_factors[name] = np.array([getter(features[s]) for s in symbols])

        if not raw_factors:
            return {}

        # ── 横截面 Z-score 归一化（含异常值截断 + 方向修正）─────────────────────
        composite = np.zeros(n)
        for factor_name, arr in raw_factors.items():
            mu    = arr.mean()
            sigma = arr.std()

            if sigma < 1e-8:
                # 所有品种该因子值相同，无区分度，跳过
                continue

            z = (arr - mu) / sigma
            z = np.clip(z, -WINSORIZE_SIGMA, WINSORIZE_SIGMA)
            sign   = FACTOR_SIGNS.get(factor_name, 1.0)
            weight = self._weights[factor_name]
            composite += weight * (sign * z)

        # ── 最终横截面归一化（使得分尺度可解释，便于阈值设定）────────────────────
        c_std = composite.std()
        if c_std > 1e-8:
            composite = (composite - composite.mean()) / c_std

        result = dict(zip(symbols, composite.tolist()))

        top5 = sorted(result.items(), key=lambda x: -x[1])[:5]
        top5_str = " | ".join(f"{s}:{v:.3f}" for s, v in top5)
        logger.debug(f"[ScoringEngine] Top5: {top5_str}")

        return result

    def get_factor_breakdown(
        self, symbol: str, features: Dict[str, SymbolFeatures]
    ) -> Optional[Dict[str, float]]:
        """
        返回单个品种的因子分解（用于调试和解释）。

        格式：{"volume_zscore": 0.35, "volume_zscore_contrib": 0.105, ..., "total": 0.72}
        """
        if symbol not in features or len(features) < MIN_SYMBOLS:
            return None

        scores = self.compute_scores(features)
        if symbol not in scores:
            return None

        symbols = list(features.keys())
        idx = symbols.index(symbol)
        breakdown = {}

        for factor_name, getter in _FACTOR_GETTERS.items():
            if self._weights.get(factor_name, 0.0) == 0.0:
                continue
            arr = np.array([getter(features[s]) for s in symbols])
            mu, sigma = arr.mean(), arr.std()
            if sigma > 1e-8:
                z = float(np.clip((arr[idx] - mu) / sigma, -WINSORIZE_SIGMA, WINSORIZE_SIGMA))
            else:
                z = 0.0
            sign     = FACTOR_SIGNS.get(factor_name, 1.0)
            z_signed = sign * z
            breakdown[factor_name]              = z_signed
            breakdown[f"{factor_name}_contrib"] = z_signed * self._weights.get(factor_name, 0.0)

        breakdown["total"] = scores[symbol]
        return breakdown

    def update_weights(self, ic_updates: Dict[str, float], alpha: float = 0.2):
        """
        基于近期因子 IC（预测力）动态调整权重（EMA 更新）。

        规则：
            有效 IC = 原始 IC × FACTOR_SIGNS（使反向因子的进化方向正确）
            IC > 0 → 预测力正 → 增权
            IC < 0 → 预测力负 → 减权
            权重约束在 [base × 0.3, base × 2.0] 之间
            更新后对全部权重做比例归一化，保持总权重不变
        """
        changed = []
        for factor, ic in ic_updates.items():
            if factor not in self._weights:
                continue
            base = FACTOR_WEIGHTS.get(factor, 0.0)
            if base == 0.0:
                continue
            old_w        = self._weights[factor]
            sign         = FACTOR_SIGNS.get(factor, 1.0)
            effective_ic = ic * sign
            adjustment   = 1.0 + alpha * max(-1.0, min(1.0, effective_ic))
            new_w        = max(base * 0.3, min(base * 2.0, old_w * adjustment))
            self._weights[factor] = new_w
            if abs(new_w - old_w) > 1e-6:
                changed.append(f"{factor}:{old_w:.3f}→{new_w:.3f}")

        active = {k: v for k, v in self._weights.items() if v > 0}
        total  = sum(active.values())
        if total > 0 and changed:
            target = sum(w for w in FACTOR_WEIGHTS.values() if w > 0)
            scale  = target / total
            for f in active:
                self._weights[f] *= scale
            logger.info(f"[ScoringEngine] 权重进化: {' | '.join(changed)}")

    def get_current_weights(self) -> Dict[str, float]:
        """返回当前运行时权重（供报告展示）"""
        return dict(self._weights)

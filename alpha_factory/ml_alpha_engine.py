"""
ML Alpha Engine - XGBoost 横截面预测引擎

职责：
    替代手写因子权重，用机器学习模型预测未来 60s 横截面相对收益，
    输出与 ScoringEngine.compute_scores() 格式完全相同的得分字典。

核心思想：
    每轮 rank_interval（60s）：
        1. 用上一轮的特征 + 当前价格，生成 log_return 标签
        2. 累积足够样本后，训练 XGBRegressor
        3. 对当前全市场特征做横截面预测 → 排名 → 替换手写得分

与 ScoringEngine 的关系：
    - 热身期（样本不足）：使用 ScoringEngine（手写加权）
    - 模型就绪后：使用 MLAlphaEngine（数据驱动）
    - 评估指标：IC（Information Coefficient = corr(pred, actual_return)）

特征体系：
    ofi, depth_imbalance, oi_change_pct, volume_zscore,
    ret_1m, ret_5m, funding_rate, spread_bps,
    lob_z1, lob_z2, lob_z3

标签：
    y = log(price[t + label_horizon] / price[t])
    横截面 de-mean：减去同截面均值（消除市场整体涨跌的共同成分）
"""

import math
import numpy as np
from collections import deque
from copy import copy
from typing import Dict, List, Optional, Tuple

from data_layer.logger import logger
from .feature_engine import SymbolFeatures


# ── 特征名称（顺序固定，不能随意更改）─────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    "ofi",
    "depth_imbalance",
    "oi_change_pct",
    "volume_zscore",
    "ret_1m",
    "ret_5m",
    "funding_rate",
    "spread_bps",
    "lob_z1",
    "lob_z2",
    "lob_z3",
]

# ── 超参数 ────────────────────────────────────────────────────────────────────
MIN_DATA_COUNT  = 30    # 品种至少接收 N 笔行情才纳入训练/预测
MIN_SYMBOLS     = 8     # 横截面至少 N 个品种才做预测
MAX_BUFFER_ROWS = 5000  # 训练数据最多保留 N 行（最近）


def _extract(feat: SymbolFeatures) -> List[float]:
    """从 SymbolFeatures 提取特征向量（顺序与 FEATURE_NAMES 对齐）"""
    return [
        feat.ofi,
        feat.depth_imbalance,
        feat.oi_change_pct,
        feat.volume_zscore,
        feat.ret_1m,
        feat.ret_5m,
        feat.funding_rate,
        feat.spread_bps,
        feat.lob_z1,
        feat.lob_z2,
        feat.lob_z3,
    ]


class MLAlphaEngine:
    """
    XGBoost 横截面收益预测引擎。

    调用方式：
        # 每个 rank_interval 调用一次
        ml_engine.update(features, now)

        # 若模型就绪，用 ML 得分替换手写得分
        if ml_engine.is_ready:
            scores = ml_engine.predict(features)
        else:
            scores = scoring_engine.compute_scores(features)

    参数说明：
        min_samples      : 触发首次训练所需的最少样本行数
        retrain_every    : 每累积多少新样本行触发一次重训
        label_horizon    : 标签预测窗口（以 rank_interval 为单位，默认 1 = 预测下一轮）
    """

    def __init__(
        self,
        min_samples:   int = 200,
        retrain_every: int = 50,
        label_horizon: int = 1,
    ):
        self.min_samples   = min_samples
        self.retrain_every = retrain_every
        self.label_horizon = label_horizon  # 单位：rank_interval 步数

        # 快照队列：保留最近 label_horizon+1 个截面快照
        # 每个快照 = {"ts": float, "feats": {sym: List[float]}, "prices": {sym: float}}
        self._snapshots: deque = deque(maxlen=label_horizon + 1)

        # 训练数据（滚动，保留最近 MAX_BUFFER_ROWS 行）
        self._X: deque = deque(maxlen=MAX_BUFFER_ROWS)
        self._y: deque = deque(maxlen=MAX_BUFFER_ROWS)

        self._model = None
        self._is_ready: bool = False
        self._samples_since_retrain: int = 0

        # 评估统计
        self.last_ic:        float = 0.0
        self.train_count:    int   = 0
        self.total_samples:  int   = 0
        self.feature_importance: Dict[str, float] = {}

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def update(self, features: Dict[str, SymbolFeatures], now: float) -> None:
        """
        每个 rank_interval 调用一次：
        1. 用上一轮快照 + 当前价格生成标签行
        2. 记录当前快照
        3. 若样本足够且达到重训间隔，触发训练

        调用后，用 is_ready 判断是否可以调用 predict()。
        """
        # ── 过滤数据不足的品种 ───────────────────────────────────────────────
        valid = {
            sym: feat for sym, feat in features.items()
            if feat.data_count >= MIN_DATA_COUNT and feat.last_price > 0
        }
        if len(valid) < MIN_SYMBOLS:
            return

        # ── 当前价格快照 ─────────────────────────────────────────────────────
        current_prices = {sym: feat.last_price for sym, feat in valid.items()}

        # ── 生成标签：用 label_horizon 步前的快照 + 当前价格 ─────────────────
        if len(self._snapshots) == self._snapshots.maxlen:
            old_snap = self._snapshots[0]
            new_rows = self._make_labels(old_snap, current_prices)
            for x_row, y_val in new_rows:
                self._X.append(x_row)
                self._y.append(y_val)
                self._samples_since_retrain += 1
                self.total_samples += 1

        # ── 记录当前快照 ─────────────────────────────────────────────────────
        self._snapshots.append({
            "ts":     now,
            "feats":  {sym: _extract(feat) for sym, feat in valid.items()},
            "prices": current_prices,
        })

        # ── 触发训练 ─────────────────────────────────────────────────────────
        n = len(self._X)
        if n >= self.min_samples and self._samples_since_retrain >= self.retrain_every:
            self._train()

        logger.debug(
            f"[MLAlpha] samples={n} since_retrain={self._samples_since_retrain} "
            f"ready={self._is_ready} IC={self.last_ic:+.4f}"
        )

    def predict(self, features: Dict[str, SymbolFeatures]) -> Dict[str, float]:
        """
        对当前截面做预测，返回横截面归一化得分。
        格式与 ScoringEngine.compute_scores() 完全相同。

        未就绪时返回空字典（调用方应降级到 ScoringEngine）。
        """
        if not self._is_ready:
            return {}

        valid = {
            sym: feat for sym, feat in features.items()
            if feat.data_count >= MIN_DATA_COUNT and feat.last_price > 0
        }
        if len(valid) < MIN_SYMBOLS:
            return {}

        symbols = list(valid.keys())
        X = np.array([_extract(valid[s]) for s in symbols], dtype=np.float32)

        # NaN/Inf 填 0（避免 XGBoost 崩溃）
        X = np.where(np.isfinite(X), X, 0.0)

        raw_preds = self._model.predict(X)  # shape: (n_symbols,)

        # ── 横截面归一化（与 ScoringEngine 对齐，使分数可解释）────────────────
        mu    = raw_preds.mean()
        sigma = raw_preds.std()
        if sigma > 1e-8:
            preds = (raw_preds - mu) / sigma
        else:
            preds = raw_preds - mu  # 方差极小时不缩放

        result = dict(zip(symbols, preds.tolist()))

        top5 = sorted(result.items(), key=lambda x: -x[1])[:5]
        top5_str = " | ".join(f"{s}:{v:+.3f}" for s, v in top5)
        logger.debug(f"[MLAlpha] 预测 Top5: {top5_str}")

        return result

    def get_status(self) -> dict:
        """返回引擎状态（供 Dashboard / 报表使用）"""
        return {
            "is_ready":          self._is_ready,
            "total_samples":     self.total_samples,
            "train_count":       self.train_count,
            "last_ic":           round(self.last_ic, 4),
            "feature_importance": self.feature_importance,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _make_labels(
        self,
        old_snap: dict,
        current_prices: Dict[str, float],
    ) -> List[Tuple[List[float], float]]:
        """
        用旧快照的特征 + 当前价格生成训练行列表。

        标签：横截面 de-meaned log return
            y_raw  = log(p_now / p_old)
            y_mean = 同截面所有品种的 y_raw 均值（消除市场整体涨跌）
            y      = y_raw - y_mean

        De-mean 是横截面 alpha 的核心：我们预测的是"相对强弱"，
        而不是绝对涨跌（绝对涨跌受 BTC 拖动，无法通过选股获得）。
        """
        old_feats  = old_snap["feats"]
        old_prices = old_snap["prices"]

        rows = []
        for sym, x_vec in old_feats.items():
            p_old = old_prices.get(sym, 0.0)
            p_new = current_prices.get(sym, 0.0)
            if p_old <= 0 or p_new <= 0:
                continue
            y_raw = math.log(p_new / p_old)
            rows.append((x_vec, y_raw, sym))

        if not rows:
            return []

        # 横截面 de-mean（消除市场公共因子）
        ys   = [r[1] for r in rows]
        mean = sum(ys) / len(ys)
        return [(r[0], r[1] - mean) for r in rows]

    def _train(self) -> None:
        """训练 XGBRegressor，时间切分评估 IC，更新模型。"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            logger.warning("[MLAlpha] XGBoost 未安装，跳过训练。pip install xgboost")
            return

        X = np.array(list(self._X), dtype=np.float32)
        y = np.array(list(self._y), dtype=np.float32)

        # 过滤 NaN / Inf
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(X) < self.min_samples:
            return

        # 时间切分（绝不 shuffle：时间序列数据）
        split   = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = XGBRegressor(
            n_estimators      = 300,
            max_depth         = 4,
            learning_rate     = 0.04,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            min_child_weight  = 5,
            reg_lambda        = 1.0,
            tree_method       = "hist",
            verbosity         = 0,
            n_jobs            = 2,
        )
        model.fit(
            X_train, y_train,
            eval_set           = [(X_val, y_val)],
            verbose            = False,
        )

        # ── IC 评估（预测与实际收益的 Pearson 相关）────────────────────────
        ic = 0.0
        if len(X_val) > 5:
            y_pred = model.predict(X_val)
            mask_v = np.isfinite(y_pred) & np.isfinite(y_val)
            if mask_v.sum() > 5:
                corr = np.corrcoef(y_pred[mask_v], y_val[mask_v])
                ic   = float(corr[0, 1]) if np.isfinite(corr[0, 1]) else 0.0

        # ── 特征重要性（方便调试）─────────────────────────────────────────
        imp = model.feature_importances_
        self.feature_importance = dict(zip(FEATURE_NAMES, imp.tolist()))

        self._model = model
        self._is_ready = True
        self._samples_since_retrain = 0
        self.last_ic   = ic
        self.train_count += 1

        imp_str = " ".join(
            f"{n}:{v:.3f}" for n, v in
            sorted(self.feature_importance.items(), key=lambda x: -x[1])[:5]
        )
        logger.info(
            f"[MLAlpha] 第{self.train_count}次训练完成 | "
            f"samples={len(X)} IC(val)={ic:+.4f} | "
            f"Top特征: {imp_str}"
        )

"""
LOB Alpha Engine — 从坐标 (z1,z2,z3) 到可交易 alpha 函数

流程：
    record_state()     每轮排序记录当前状态
    on_price_update()  每次 Tick 填入未来收益标签
    maybe_fit()        满足最少样本后自动拟合
    score_z()          实时预测，集成进 ScoringEngine

模型层次：
    Level 0 (fallback)   线性，热身期使用
    Level 1 (基础)       单变量加法分段：f(z) = T₁(z₁) + T₂(z₂) + T₃(z₃)
                         T_k 由分位数桶内均值收益估计
    Level 2 (增强)       双变量交互矫正：residual[b₁, b₂] = E[R|bin₁,bin₂] - additive
    Level 3 (条件化)     按 regime × bucket 独立模型，数据不足时降级到 Level 1

研究工具（供 Dashboard / 离线分析使用）：
    bucket_analysis()      单变量分桶收益
    conditional_ic_table() IC × regime × bucket
    bivariate_surface()    双变量收益热力图
    export_analysis()      写出 CSV
"""

import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_layer.logger import logger


# ── 超参数 ─────────────────────────────────────────────────────────────────────

HORIZON_S        = 60.0    # 预测时间窗（秒）：记录时刻的收益将在此时间后被标注
MIN_OBS          = 200     # 触发首次拟合所需的最少已标注样本数
RETRAIN_EVERY    = 50      # 每积累此数量新样本后重新拟合
MAX_BUFFER       = 3000    # 已标注样本滚动缓冲区大小
MAX_PENDING_S    = 300.0   # 超过此时间未收到价格更新的 pending 样本直接丢弃

N_BINS_MAIN      = 10      # 单变量分桶数（Level 1）
N_BINS_BIVAR     = 5       # 双变量分桶数（Level 2，5×5 格）
MIN_BIN_SAMPLES  = 5       # 单个桶最少样本（少于此不估计，置 0）

# Level 2 交互项权重（0 = 纯加法；1 = 全用交互矫正）
BIVAR_WEIGHT     = 0.3


# ── 数据结构 ───────────────────────────────────────────────────────────────────

@dataclass
class _Obs:
    symbol:        str
    z:             np.ndarray   # shape (3,)，白化后的 lob_z1/z2/z3
    bucket:        str          # "high" / "mid" / "low"
    is_resonance:  bool         # structure_regime
    is_shock:      bool         # shock_regime
    price:         float        # 记录时刻价格
    ts:            float        # 记录时刻时间戳（Unix s）
    ret_label:     float = 0.0  # 未来 HORIZON_S 秒收益，由 on_price_update 填入
    labeled:       bool  = False


# ── 分段加法模型 ───────────────────────────────────────────────────────────────

class _AdditiveModel:
    """
    加法分段模型：f(z) = Σ_k T_k(z_k)

    T_k 通过对 z_k 做分位数分桶、取桶内均值收益来估计。
    预测时按 z_k 落入的桶直接查表，O(log N) 单次预测。
    """

    def __init__(self, n_bins: int = N_BINS_MAIN):
        self.n_bins   = n_bins
        self._edges:  Optional[np.ndarray] = None   # (3, n_bins+1)
        self._values: Optional[np.ndarray] = None   # (3, n_bins)
        self._fitted  = False

    def fit(self, Z: np.ndarray, R: np.ndarray):
        """Z: (N,3)  R: (N,)"""
        n_dims = Z.shape[1]
        edges  = np.zeros((n_dims, self.n_bins + 1))
        values = np.zeros((n_dims, self.n_bins))

        for k in range(n_dims):
            z_k   = Z[:, k]
            q     = np.linspace(0, 1, self.n_bins + 1)
            e     = np.quantile(z_k, q)
            edges[k] = e
            for j in range(self.n_bins):
                lo, hi = e[j], e[j + 1]
                mask = (z_k >= lo) & (z_k <= hi) if j == self.n_bins - 1 \
                       else (z_k >= lo) & (z_k < hi)
                if mask.sum() >= MIN_BIN_SAMPLES:
                    values[k, j] = R[mask].mean()

        self._edges  = edges
        self._values = values
        self._fitted = True

    def predict(self, z: np.ndarray) -> float:
        """z: (3,) → scalar score"""
        if not self._fitted:
            return float(z.mean())   # 均值 fallback
        score = 0.0
        for k in range(3):
            j = int(np.searchsorted(self._edges[k], z[k], side="right")) - 1
            j = max(0, min(j, self.n_bins - 1))
            score += self._values[k, j]
        return score

    def predict_batch(self, Z: np.ndarray) -> np.ndarray:
        return np.array([self.predict(z) for z in Z])

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def bucket_profile(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """返回指定维度的分桶中心值和对应均值收益（供 research 使用）。"""
        if not self._fitted:
            return np.array([]), np.array([])
        centers = (self._edges[dim, :-1] + self._edges[dim, 1:]) / 2
        return centers, self._values[dim]


# ── 主引擎 ────────────────────────────────────────────────────────────────────

class LOBAlphaEngine:
    """
    在线 LOB Alpha 学习引擎。

    数据流：
      1. record_state()    每 60s 排序时，对所有品种记录 (z, regime, bucket, price)
      2. on_price_update() 每次 Tick 触发，扫描 pending 中到期样本，填入收益标签
      3. maybe_fit()       积累够 MIN_OBS 后自动拟合，后续每 RETRAIN_EVERY 重拟合
      4. score_z()         实时预测，在 ScoringEngine 中替换线性 lob_z 权重

    模型选择：
      - 若当前 regime = is_resonance → 使用对应桶模型（若不够数据则降级到全局）
      - 若 is_shock → 直接返回 0（当前冲击时不做 lob 贡献）
    """

    def __init__(
        self,
        horizon_s:    float = HORIZON_S,
        min_obs:      int   = MIN_OBS,
        retrain_every: int  = RETRAIN_EVERY,
        export_dir:   str   = ".",
    ):
        self._horizon     = horizon_s
        self._min_obs     = min_obs
        self._retrain_every = retrain_every
        self._export_dir  = export_dir

        # pending[symbol] = list of unlabeled _Obs
        self._pending: Dict[str, List[_Obs]] = {}
        # completed: rolling buffer of labeled observations
        self._completed: deque = deque(maxlen=MAX_BUFFER)

        self._n_new_labeled   = 0   # 上次拟合后新增标注数
        self._total_labeled   = 0   # 累计标注总数

        # 模型：global + per-(resonance, bucket)
        # key = "global" | "res_{bool}_bkt_{str}"
        self._models: Dict[str, _AdditiveModel] = {}

        # Level 2：双变量交互矫正（全局一套）
        # _bivar_correction[b1, b2] = mean_residual
        self._bivar_edges:      Optional[np.ndarray] = None  # (2, N_BINS_BIVAR+1)
        self._bivar_correction: Optional[np.ndarray] = None  # (N_BINS_BIVAR, N_BINS_BIVAR)

        self._is_fitted = False

    # ── 数据接入 ──────────────────────────────────────────────────────────────

    def record_state(
        self,
        symbol:       str,
        z:            np.ndarray,
        bucket:       str,
        regime_state: dict,
        price:        float,
        ts:           Optional[float] = None,
    ):
        """
        每轮排序时调用，记录当前 (z, regime, bucket, price)。

        regime_state 来自 LOBManifoldEngine.get_regime_state()：
          regime_state["structure_regime"]["is_resonance"]
          regime_state["shock_regime"]["is_shock"]
        """
        if price <= 0 or z is None or len(z) < 3:
            return

        now = ts or time.time()
        s   = regime_state.get("structure_regime", {})
        sh  = regime_state.get("shock_regime", {})

        obs = _Obs(
            symbol       = symbol,
            z            = z.copy(),
            bucket       = bucket,
            is_resonance = bool(s.get("is_resonance", False)),
            is_shock     = bool(sh.get("is_shock", False)),
            price        = price,
            ts           = now,
        )
        if symbol not in self._pending:
            self._pending[symbol] = []
        self._pending[symbol].append(obs)

    def on_price_update(self, symbol: str, price: float, ts: Optional[float] = None):
        """
        每次 Tick 时调用，扫描该品种的 pending 列表：
          - 到期（ts_obs + horizon <= now）→ 计算收益标签，移入 completed
          - 过期太久（> MAX_PENDING_S）→ 直接丢弃
        """
        if price <= 0 or symbol not in self._pending:
            return

        now     = ts or time.time()
        pending = self._pending[symbol]
        remaining = []

        for obs in pending:
            age = now - obs.ts
            if age >= self._horizon:
                obs.ret_label = (price - obs.price) / (obs.price + 1e-10)
                obs.labeled   = True
                self._completed.append(obs)
                self._n_new_labeled  += 1
                self._total_labeled  += 1
            elif age > MAX_PENDING_S:
                pass   # 过期丢弃
            else:
                remaining.append(obs)

        self._pending[symbol] = remaining

    # ── 模型拟合 ──────────────────────────────────────────────────────────────

    def maybe_fit(self):
        """
        在排序循环结束后调用。
        满足条件时触发拟合：
          - 总标注数 >= MIN_OBS  且
          - 上次拟合后新增 >= RETRAIN_EVERY
        """
        if (self._total_labeled >= self._min_obs
                and self._n_new_labeled >= self._retrain_every):
            self._fit()
            self._n_new_labeled = 0

    def _fit(self):
        obs_list = list(self._completed)
        if len(obs_list) < self._min_obs:
            return

        Z   = np.array([o.z        for o in obs_list])   # (N, 3)
        R   = np.array([o.ret_label for o in obs_list])  # (N,)

        # ── Level 1：全局加法模型 ─────────────────────────────────────────────
        global_model = _AdditiveModel(N_BINS_MAIN)
        global_model.fit(Z, R)
        self._models["global"] = global_model

        # ── Level 1：条件化模型（resonance × bucket）─────────────────────────
        groups = {}
        for obs in obs_list:
            key = f"res_{obs.is_resonance}_bkt_{obs.bucket}"
            groups.setdefault(key, []).append(obs)

        for key, group in groups.items():
            if len(group) >= self._min_obs // 4:
                Z_g = np.array([o.z for o in group])
                R_g = np.array([o.ret_label for o in group])
                m   = _AdditiveModel(N_BINS_MAIN)
                m.fit(Z_g, R_g)
                self._models[key] = m

        # ── Level 2：双变量交互矫正（z1 × z2，基于全局样本）────────────────
        self._fit_bivariate(Z, R, global_model)

        self._is_fitted = True
        logger.info(
            f"[LOBAlphaEngine] 拟合完成 N={len(obs_list)} "
            f"conditions={list(self._models.keys())} "
            f"bivar={'yes' if self._bivar_correction is not None else 'no'}"
        )

    def _fit_bivariate(
        self,
        Z: np.ndarray,
        R: np.ndarray,
        global_model: _AdditiveModel,
    ):
        """
        Level 2：估计双变量交互矫正项。

        residual[b1, b2] = E[R - additive_pred | z1_bin=b1, z2_bin=b2]

        仅用 z1 × z2（最强两个因子），避免样本稀疏。
        """
        if len(Z) < N_BINS_BIVAR ** 2 * MIN_BIN_SAMPLES:
            return

        z1, z2 = Z[:, 0], Z[:, 1]
        q      = np.linspace(0, 1, N_BINS_BIVAR + 1)
        e1     = np.quantile(z1, q)
        e2     = np.quantile(z2, q)

        additive_pred = global_model.predict_batch(Z)
        residual      = R - additive_pred

        correction = np.zeros((N_BINS_BIVAR, N_BINS_BIVAR))
        for b1 in range(N_BINS_BIVAR):
            for b2 in range(N_BINS_BIVAR):
                m1 = (z1 >= e1[b1]) & (z1 <= e1[b1 + 1] if b1 == N_BINS_BIVAR - 1 else z1 < e1[b1 + 1])
                m2 = (z2 >= e2[b2]) & (z2 <= e2[b2 + 1] if b2 == N_BINS_BIVAR - 1 else z2 < e2[b2 + 1])
                mask = m1 & m2
                if mask.sum() >= MIN_BIN_SAMPLES:
                    correction[b1, b2] = residual[mask].mean()

        self._bivar_edges      = np.array([e1, e2])
        self._bivar_correction = correction

    # ── 实时预测 ──────────────────────────────────────────────────────────────

    def score_z(
        self,
        z:            np.ndarray,
        bucket:       str,
        regime_state: dict,
    ) -> float:
        """
        给定单个品种的 z 坐标，返回 lob_alpha_score。

        shock 状态下直接返回 0（不参与开仓）。
        高共振时使用对应条件模型（数据不足则降级到全局）。
        """
        sh = regime_state.get("shock_regime", {})
        if sh.get("is_shock", False):
            return 0.0

        s   = regime_state.get("structure_regime", {})
        res = bool(s.get("is_resonance", False))

        # 选择最适合的模型
        key     = f"res_{res}_bkt_{bucket}"
        model   = self._models.get(key) or self._models.get("global")
        if model is None or not model.is_fitted:
            # 纯线性 fallback（热身期）
            return float(np.dot(z, [0.30, 0.40, 0.30]))

        additive = model.predict(z)

        # Level 2：加入双变量交互矫正
        bivar_delta = 0.0
        if self._bivar_correction is not None and self._bivar_edges is not None:
            b1 = int(np.searchsorted(self._bivar_edges[0], z[0], side="right")) - 1
            b2 = int(np.searchsorted(self._bivar_edges[1], z[1], side="right")) - 1
            b1 = max(0, min(b1, N_BINS_BIVAR - 1))
            b2 = max(0, min(b2, N_BINS_BIVAR - 1))
            bivar_delta = float(self._bivar_correction[b1, b2])

        return (1.0 - BIVAR_WEIGHT) * additive + BIVAR_WEIGHT * (additive + bivar_delta)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def total_labeled(self) -> int:
        return self._total_labeled

    # ── 研究工具 ──────────────────────────────────────────────────────────────

    def bucket_analysis(
        self,
        z_idx:         int  = 1,
        n_bins:        int  = 10,
        regime_filter: Optional[str] = None,
        bucket_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        单变量分桶收益分析。

        参数：
            z_idx         0/1/2 对应 z1/z2/z3
            n_bins        分桶数
            regime_filter "resonance" / "no_resonance" / None（不过滤）
            bucket_filter "high" / "mid" / "low" / None

        返回：
            list of {"bin_center": float, "mean_ret": float, "n": int, "sharpe": float}
        """
        obs_list = self._filter_obs(regime_filter, bucket_filter)
        if len(obs_list) < n_bins * MIN_BIN_SAMPLES:
            return []

        zk = np.array([o.z[z_idx] for o in obs_list])
        R  = np.array([o.ret_label for o in obs_list])

        edges   = np.quantile(zk, np.linspace(0, 1, n_bins + 1))
        results = []
        for j in range(n_bins):
            lo, hi = edges[j], edges[j + 1]
            mask = (zk >= lo) & (zk <= hi) if j == n_bins - 1 else (zk >= lo) & (zk < hi)
            if mask.sum() < MIN_BIN_SAMPLES:
                continue
            r_bin  = R[mask]
            sharpe = float(r_bin.mean() / (r_bin.std() + 1e-10)) * math.sqrt(252 * 24)
            results.append({
                "bin_center": float((lo + hi) / 2),
                "mean_ret":   float(r_bin.mean()),
                "std_ret":    float(r_bin.std()),
                "n":          int(mask.sum()),
                "sharpe":     round(sharpe, 3),
            })
        return results

    def conditional_ic_table(self) -> List[dict]:
        """
        按 regime × bucket × z_dim 计算 Spearman IC。

        返回：
            list of {"z_dim": int, "regime": str, "bucket": str, "ic": float, "n": int}
        """
        rows = []
        for res in [False, True]:
            for bkt in ["high", "mid", "low"]:
                tag = f"res_{res}_bkt_{bkt}"
                obs = [o for o in self._completed
                       if o.is_resonance == res and o.bucket == bkt]
                if len(obs) < 20:
                    continue
                Z = np.array([o.z for o in obs])
                R = np.array([o.ret_label for o in obs])
                for k in range(3):
                    ic = self._spearman_ic(Z[:, k], R)
                    rows.append({
                        "z_dim":   k + 1,
                        "regime":  "resonance" if res else "normal",
                        "bucket":  bkt,
                        "ic":      round(ic, 4),
                        "n":       len(obs),
                    })
        return rows

    def bivariate_surface(self, n_bins: int = 5) -> dict:
        """
        z1 × z2 双变量收益热力图。

        返回：
            {
              "edges_z1": [...],  # (n_bins+1,)
              "edges_z2": [...],
              "mean_ret": [[...]] # (n_bins, n_bins)
              "n_obs":    [[...]]
            }
        """
        obs_list = list(self._completed)
        if len(obs_list) < n_bins ** 2 * MIN_BIN_SAMPLES:
            return {}

        Z  = np.array([o.z for o in obs_list])
        R  = np.array([o.ret_label for o in obs_list])
        q  = np.linspace(0, 1, n_bins + 1)
        e1 = np.quantile(Z[:, 0], q)
        e2 = np.quantile(Z[:, 1], q)

        mean_ret = np.zeros((n_bins, n_bins))
        n_obs    = np.zeros((n_bins, n_bins), dtype=int)

        for b1 in range(n_bins):
            for b2 in range(n_bins):
                m1 = (Z[:, 0] >= e1[b1]) & (Z[:, 0] <= e1[b1+1] if b1==n_bins-1 else Z[:, 0] < e1[b1+1])
                m2 = (Z[:, 1] >= e2[b2]) & (Z[:, 1] <= e2[b2+1] if b2==n_bins-1 else Z[:, 1] < e2[b2+1])
                mask = m1 & m2
                n_obs[b1, b2]    = int(mask.sum())
                if mask.sum() >= MIN_BIN_SAMPLES:
                    mean_ret[b1, b2] = float(R[mask].mean())

        return {
            "edges_z1": e1.tolist(),
            "edges_z2": e2.tolist(),
            "mean_ret": mean_ret.tolist(),
            "n_obs":    n_obs.tolist(),
        }

    def export_analysis(self, path: Optional[str] = None):
        """
        把全部研究结果写出为 CSV 文件（供 Dashboard / 离线分析使用）：
          lob_bucket_z1.csv / lob_bucket_z2.csv / lob_bucket_z3.csv
          lob_ic_table.csv
          lob_surface.csv
        """
        out = path or self._export_dir
        os.makedirs(out, exist_ok=True)

        # 单变量分桶
        for z_idx in range(3):
            rows = self.bucket_analysis(z_idx=z_idx, n_bins=10)
            if not rows:
                continue
            fpath = os.path.join(out, f"lob_bucket_z{z_idx+1}.csv")
            with open(fpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

        # IC 表
        ic_rows = self.conditional_ic_table()
        if ic_rows:
            fpath = os.path.join(out, "lob_ic_table.csv")
            with open(fpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ic_rows[0].keys())
                w.writeheader()
                w.writerows(ic_rows)

        # 双变量热力图（展平写入）
        surface = self.bivariate_surface()
        if surface:
            mr    = np.array(surface["mean_ret"])
            no    = np.array(surface["n_obs"])
            fpath = os.path.join(out, "lob_surface.csv")
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["z1_bin", "z2_bin", "mean_ret", "n_obs"])
                for b1 in range(mr.shape[0]):
                    for b2 in range(mr.shape[1]):
                        w.writerow([b1, b2, round(mr[b1, b2], 6), int(no[b1, b2])])

        logger.info(f"[LOBAlphaEngine] 研究报告已导出至 {out}")

    def get_status(self) -> dict:
        return {
            "is_fitted":      self._is_fitted,
            "total_labeled":  self._total_labeled,
            "n_pending_syms": len(self._pending),
            "n_pending_obs":  sum(len(v) for v in self._pending.values()),
            "n_completed":    len(self._completed),
            "models":         list(self._models.keys()),
        }

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _filter_obs(
        self,
        regime_filter: Optional[str],
        bucket_filter: Optional[str],
    ) -> List[_Obs]:
        obs_list = list(self._completed)
        if regime_filter == "resonance":
            obs_list = [o for o in obs_list if o.is_resonance]
        elif regime_filter == "no_resonance":
            obs_list = [o for o in obs_list if not o.is_resonance]
        if bucket_filter:
            obs_list = [o for o in obs_list if o.bucket == bucket_filter]
        return obs_list

    @staticmethod
    def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
        """Spearman 秩相关系数（无需 scipy）。"""
        if len(x) < 4:
            return 0.0
        rank_x = np.argsort(np.argsort(x)).astype(float)
        rank_y = np.argsort(np.argsort(y)).astype(float)
        rx, ry = rank_x - rank_x.mean(), rank_y - rank_y.mean()
        denom  = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / denom) if denom > 0 else 0.0

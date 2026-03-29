"""
LOB Manifold Engine v5 — 统计严格版（二阶修复）

v5 在 v4 基础上的 6 项必改修复：

必改1：分桶标准化（替代全局标准化）
  问题：全局 μ/σ 混入了组间差异，桶内协方差学到的不是"桶内结构"。
  修复：_feat_mean/_feat_var 移入 _BucketCovState，标准化口径与协方差口径一致。
  数学：Cov(D_g⁻¹(X-μ_g)|g) 才是纯桶内相关结构；全局标准化得到 D⁻¹Σ_g D⁻¹。

必改2：variance_ratio 分母改成 trace(Σ)
  问题：原来分母 = Σλ_k（只取前4个特征值），高估了 PC1 占比，误判共振 regime。
  修复：total_var = trace(cov_work)，等于所有特征值之和。
  数学：ρ₁ = λ₁ / tr(Σ) 才是真正的"PC1 解释方差占比"。

必改3：幂迭代 + deflation 改成 np.linalg.eigh
  问题：deflation 误差会传导；特征值接近时收敛慢；D=20 无需省常数。
  修复：np.linalg.eigh 对实对称矩阵直接做完整特征分解，稳定且精确。
  数学：Σ = UΛUᵀ，eigh 直接给出全部特征值/向量，无误差传导。

必改4：桶切换迟滞（防止坐标系突变）
  问题：事件率微小波动导致桶切换，z1/z2/z3 跳变是 ΔU^T x 伪信号。
  修复：新桶需连续 BUCKET_HYSTERESIS_S 秒确认才切换。
  数学：Δz = Uᵀ Δx（真实信号）+ ΔUᵀ x（坐标系切换噪声），迟滞消除第二项。

必改5：特征提取时序一致性（eigen_ts + 时间阈值触发）
  问题：仅靠样本计数触发，regime 快速切换时基底可能严重滞后。
  修复：记录 eigen_ts，增加时间阈值触发（MAX_EIGEN_AGE_S），get_status 暴露基底年龄。

必改6：regime 信号彻底解耦
  问题：structure_regime（协方差谱形状）和 shock_regime（当前位置偏移）混为一个信号。
  修复：get_regime_state() 返回两个独立子字典，上层可分别使用。
  数学：variance_ratio = λ₁/tr(Σ) 是结构性；avg_pc1_abs = E|u₁ᵀx|/√λ₁ 是位置性。

完整 Pipeline v5：
    1. 标准化 LOB → 无量纲向量
    2. EMA 平滑 LOB → 稳健 velocity
    3. 固定时间窗聚合（1s）
    4. 分桶（含迟滞保护）
    5. 桶内标准化（μ_g, σ_g）
    6. 分桶 EWMA 协方差更新
    7. eigh 特征分解 + shrinkage + 符号锚定（样本数 + 时间双阈值触发）
    8. PC1 保留为 structure/shock regime 指示
    9. 白化投影（z̃_k = z_k/√λ_k）+ EMA 平滑 → lob_z1/z2/z3
"""

import math
import threading
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional

from data_layer.logger import logger


# ── 超参数 ─────────────────────────────────────────────────────────────────────

LOB_LEVELS           = 5       # 每侧取 top K 档，向量维度 = 4K
PCA_COMPONENTS       = 3       # 输出 lob_z1/z2/z3
EWMA_HALF_LIFE_S     = 60.0    # EWMA 协方差半衰期（秒）
EIGEN_UPDATE_INTERVAL = 20     # 每 N 个新样本触发一次特征提取（样本计数阈值）
MAX_EIGEN_AGE_S      = 120.0   # [必改5] 基底超过此年龄也强制重提（时间阈值）
WARMUP_SYMBOLS       = 15      # 热身最少品种数
MIN_UPDATE_INTERVAL_S = 0.05   # 单品种最小更新间隔（限速）

AGG_WINDOW_S         = 1.0     # 固定时间窗聚合宽度（秒）
AGG_MIN_SAMPLES      = 2       # 时间窗内最少样本数
LOB_SMOOTH_ALPHA     = 0.4     # LOB EMA 平滑系数（velocity 降噪）
FEAT_STD_HALFLIFE_S  = 120.0   # [必改1] 桶内标准化 EWMA 半衰期
SHRINKAGE_ETA        = 0.1     # 协方差 shrinkage：Σ_shrunk=(1-η)Σ+η·diag(Σ)
LATENT_EMA_ALPHA     = 0.25    # 输出 lob_z EMA 平滑系数

# [必改4] 桶切换迟滞：新桶需连续此时长才切换
BUCKET_HYSTERESIS_S  = 5.0

# [必改2] PC1 方差占比阈值（基于 trace，不再是前 N 特征值之和）
REGIME_RESONANCE_THRESHOLD = 0.50

# [必改6] PC1 位置性冲击阈值：avg|z̃₁| 超过此值认为市场整体偏移显著
SHOCK_PC1_THRESHOLD  = 2.0

LIQUIDITY_BUCKETS = [
    ("high", 5.0, float("inf")),   # ≥5 events/s：大币
    ("mid",  1.0, 5.0),            # 1~5 events/s
    ("low",  0.0, 1.0),            # <1 event/s：小币
]


# ── 分桶状态 ───────────────────────────────────────────────────────────────────

class _BucketCovState:
    """
    单个流动性桶的完整状态：
      - 桶内 EWMA 标准化（必改1：独立 μ_g/σ_g）
      - EWMA 协方差
      - 特征向量 + 特征值
      - 基底版本号与时间戳（必改5）
    """

    def __init__(self, D: int, n_comp: int, tau: float, feat_tau: float):
        self.D        = D
        self.n_comp   = n_comp
        self.tau      = tau           # 协方差 EWMA 时间常数
        self.feat_tau = feat_tau      # 标准化 EWMA 时间常数（必改1）

        # ── [必改1] 桶内标准化状态（主线程独占读写，无需加锁）──────────────
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_var:  Optional[np.ndarray] = None
        self.feat_ts:   float = 0.0

        # ── EWMA 协方差（主线程写，提取线程读快照）────────────────────────
        self.cov:       Optional[np.ndarray] = None
        self.mean:      Optional[np.ndarray] = None
        self.last_ts:   float = 0.0
        self.n_samples: int   = 0

        # ── 特征向量（提取线程写，主线程投影时读，需加锁）────────────────
        self.components:          Optional[np.ndarray] = None   # (n_comp, D)
        self.component_eigenvals: Optional[np.ndarray] = None   # (n_comp,)
        self.pc1_vec:             Optional[np.ndarray] = None   # (D,)
        self.pc1_eigenval:        float = 1.0
        self.variance_ratio:      float = 0.0   # [必改2] 基于 trace(Σ)
        self.fitted:              bool  = False

        # ── [必改5] 时序一致性 ───────────────────────────────────────────
        self.eigen_ts:      float = 0.0   # 最近一次特征提取完成的时间戳
        self.eigen_version: int   = 0     # 单调递增版本号

        self.lock = threading.Lock()
        self.eigen_in_progress = False
        self.samples_since_eigen = 0

    def standardize(self, x: np.ndarray, now: float) -> np.ndarray:
        """
        [必改1] 桶内 EWMA 标准化。

        与全局标准化的差别：均值和方差估计只基于本桶样本，
        消除了"大币均值/方差污染小币协方差"的问题。
        """
        if self.feat_mean is None:
            self.feat_mean = x.copy()
            self.feat_var  = np.full(len(x), 1e-4)
            self.feat_ts   = now
            return np.zeros(len(x))

        dt    = max(0.0, now - self.feat_ts)
        decay = math.exp(-dt / self.feat_tau) if self.feat_tau > 0 else 0.0

        diff           = x - self.feat_mean
        self.feat_var  = decay * self.feat_var  + (1.0 - decay) * diff ** 2
        self.feat_mean = decay * self.feat_mean + (1.0 - decay) * x
        self.feat_ts   = now

        return diff / (np.sqrt(self.feat_var) + 1e-8)

    def update_cov(self, x: np.ndarray, now: float):
        """时间感知 EWMA 协方差更新（仅主线程调用）。"""
        if self.cov is None:
            self.cov     = np.zeros((self.D, self.D))
            self.mean    = x.copy()
            self.last_ts = now
            self.n_samples = 1
            return

        dt    = max(0.0, now - self.last_ts)
        decay = math.exp(-dt / self.tau) if self.tau > 0 else 0.0

        centered     = x - self.mean
        self.cov     = decay * self.cov + (1.0 - decay) * np.outer(centered, centered)
        self.mean    = decay * self.mean + (1.0 - decay) * x
        self.last_ts = now
        self.n_samples += 1


class LOBManifoldEngine:
    """
    LOB 流形引擎 v5：统计严格版（二阶修复）

    对外接口与 v4 完全兼容：
        on_order_book(symbol, bids, asks, mid, ts)
        get_symbol_latent(symbol) → Optional[np.ndarray]
        get_regime_state()        → dict（结构/位置 regime 解耦）
        is_ready                  → bool
    """

    def __init__(
        self,
        lob_levels:     int   = LOB_LEVELS,
        n_components:   int   = PCA_COMPONENTS,
        ema_alpha:      float = LATENT_EMA_ALPHA,
        half_life_s:    float = EWMA_HALF_LIFE_S,
        agg_window_s:   float = AGG_WINDOW_S,
        shrinkage_eta:  float = SHRINKAGE_ETA,
    ):
        self._K          = lob_levels
        self._D          = lob_levels * 4
        self._n_comp     = n_components
        self._z_ema      = ema_alpha
        self._tau        = half_life_s / math.log(2.0)
        self._agg_window = agg_window_s
        self._eta        = shrinkage_eta
        self._feat_tau   = FEAT_STD_HALFLIFE_S / math.log(2.0)

        # 每品种 LOB 平滑状态（稳健 velocity）
        self._lob_ema:  Dict[str, np.ndarray] = {}
        self._prev_ts:  Dict[str, float]      = {}

        # 每品种固定时间窗 velocity 缓冲：(timestamp, velocity_vec)
        self._delta_history: Dict[str, deque] = {}

        # ── [必改1] 分桶状态（含桶内标准化）────────────────────────────────
        self._buckets: Dict[str, _BucketCovState] = {
            name: _BucketCovState(self._D, n_components, self._tau, self._feat_tau)
            for name, _, _ in LIQUIDITY_BUCKETS
        }

        # ── [必改4] 桶切换迟滞状态 ──────────────────────────────────────────
        self._sym_bucket:           Dict[str, str]   = {}   # 当前生效桶
        self._sym_bucket_candidate: Dict[str, str]   = {}   # 候选新桶
        self._sym_candidate_since:  Dict[str, float] = {}   # 候选开始时间

        # 事件频率 EWMA（用于计算原始桶）
        self._sym_event_rate: Dict[str, float] = {}
        self._sym_last_event: Dict[str, float] = {}

        # 每品种输出
        self._lob_latent:  Dict[str, np.ndarray] = {}
        self._pc1_scores:  Dict[str, float]      = {}   # 白化 PC1 分数

        # 热身
        self._symbols_seen:      set = set()
        self._total_agg_samples: int = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def on_order_book(
        self,
        symbol: str,
        bids:   List,
        asks:   List,
        mid:    float,
        ts:     Optional[float] = None,
    ):
        if mid <= 0 or not bids or not asks:
            return

        now = ts if (ts and ts > 1e9) else time.time()

        # 限速
        if symbol in self._prev_ts:
            elapsed = now - self._prev_ts[symbol]
            if elapsed < MIN_UPDATE_INTERVAL_S:
                return
        else:
            elapsed = 0.0

        # Step 1：标准化 LOB
        lob_vec = self._normalize_lob(bids, asks, mid)
        if lob_vec is None:
            return

        # Step 2-3：EMA 平滑 + velocity
        if symbol not in self._lob_ema:
            self._lob_ema[symbol] = lob_vec.copy()
            self._prev_ts[symbol] = now
            return

        lob_smooth = LOB_SMOOTH_ALPHA * lob_vec + (1.0 - LOB_SMOOTH_ALPHA) * self._lob_ema[symbol]
        dt         = max(0.05, min(elapsed, 10.0))
        velocity   = (lob_smooth - self._lob_ema[symbol]) / dt

        self._lob_ema[symbol] = lob_smooth
        self._prev_ts[symbol] = now

        # Step 4：固定时间窗聚合
        if symbol not in self._delta_history:
            self._delta_history[symbol] = deque(maxlen=50)
        self._delta_history[symbol].append((now, velocity))

        cutoff = now - self._agg_window
        recent = [(t, v) for t, v in self._delta_history[symbol] if t >= cutoff]
        if len(recent) < AGG_MIN_SAMPLES:
            return

        x_raw = np.mean([v for _, v in recent], axis=0).astype(np.float64)

        # Step 5：[必改4] 分桶（含迟滞保护）
        bucket_name = self._get_bucket(symbol, now)
        bucket      = self._buckets[bucket_name]

        # Step 6：[必改1] 桶内标准化（口径与协方差一致）
        x = bucket.standardize(x_raw, now)

        # Step 7：更新分桶 EWMA 协方差
        bucket.update_cov(x, now)
        self._symbols_seen.add(symbol)
        self._total_agg_samples  += 1
        bucket.samples_since_eigen += 1

        # Step 8：[必改5] 样本计数 + 时间双阈值触发特征提取
        warmup_ok = (
            len(self._symbols_seen) >= WARMUP_SYMBOLS
            and self._total_agg_samples >= WARMUP_SYMBOLS * EIGEN_UPDATE_INTERVAL
        )
        eigen_age = now - bucket.eigen_ts if bucket.fitted else float("inf")
        time_trigger   = eigen_age >= MAX_EIGEN_AGE_S
        sample_trigger = bucket.samples_since_eigen >= EIGEN_UPDATE_INTERVAL

        if warmup_ok and (sample_trigger or time_trigger) and not bucket.eigen_in_progress:
            bucket.samples_since_eigen = 0
            bucket.eigen_in_progress   = True
            threading.Thread(
                target=self._extract_eigenvectors,
                args=(bucket,),
                daemon=True,
            ).start()

        # Step 9：白化投影 + EMA 平滑
        if bucket.fitted:
            self._project_and_smooth(symbol, x, bucket)

    def get_symbol_latent(self, symbol: str) -> Optional[np.ndarray]:
        return self._lob_latent.get(symbol)

    def get_all_latents(self) -> Dict[str, np.ndarray]:
        return dict(self._lob_latent)

    def get_symbol_bucket(self, symbol: str) -> str:
        """返回品种当前所属的流动性桶（含迟滞），未分配时返回 'mid'。"""
        return self._sym_bucket.get(symbol, "mid")

    def get_regime_state(self) -> dict:
        """
        [必改6] 结构性 regime 与位置性 regime 彻底解耦。

        structure_regime（协方差谱形状）：
            pc1_variance_ratio = λ₁ / tr(Σ)  — PC1 解释了多少总方差
            is_resonance       — 是否共振主导（idiosyncratic alpha 可信度下降）
            by_bucket          — 各桶独立状态 + 基底年龄

        shock_regime（当前样本位置偏移）：
            avg_pc1_abs        — 全市场白化 PC1 分数均值绝对值
            p95_pc1_abs        — 95 分位（捕捉局部冲击）
            is_shock           — 是否处于冲击状态（当前整体偏离PC1方向）

        两者上层用途不同：
            structure_regime → 控制"是否信任因子"（高共振时降低 lob_z 权重）
            shock_regime     → 控制"是否在当前时刻避开冲击"（高冲击时拒绝开仓）
        """
        now = time.time()
        structure_by_bucket = {}
        max_ratio = 0.0

        for name, bucket in self._buckets.items():
            with bucket.lock:
                ratio   = bucket.variance_ratio if bucket.fitted else 0.0
                version = bucket.eigen_version
                age     = now - bucket.eigen_ts if bucket.fitted else -1.0
            structure_by_bucket[name] = {
                "pc1_variance_ratio": round(ratio, 3),
                "is_resonance":       ratio > REGIME_RESONANCE_THRESHOLD,
                "eigen_version":      version,
                "eigen_age_s":        round(age, 1),
            }
            max_ratio = max(max_ratio, ratio)

        pc1_vals = list(self._pc1_scores.values())
        if pc1_vals:
            abs_vals = np.abs(pc1_vals)
            avg_pc1  = float(np.mean(abs_vals))
            p95_pc1  = float(np.percentile(abs_vals, 95)) if len(abs_vals) >= 5 else avg_pc1
        else:
            avg_pc1 = p95_pc1 = 0.0

        return {
            "structure_regime": {
                "max_pc1_variance_ratio": round(max_ratio, 3),
                "is_resonance":           max_ratio > REGIME_RESONANCE_THRESHOLD,
                "by_bucket":              structure_by_bucket,
            },
            "shock_regime": {
                "avg_pc1_abs": round(avg_pc1, 3),
                "p95_pc1_abs": round(p95_pc1, 3),
                "is_shock":    avg_pc1 > SHOCK_PC1_THRESHOLD,
            },
        }

    @property
    def is_ready(self) -> bool:
        return any(b.fitted for b in self._buckets.values())

    def get_status(self) -> dict:
        now = time.time()
        bucket_info = {}
        for name, b in self._buckets.items():
            with b.lock:
                fitted  = b.fitted
                version = b.eigen_version
                age     = round(now - b.eigen_ts, 1) if fitted else -1.0
            bucket_info[name] = {
                "n_samples": b.n_samples,
                "fitted":    fitted,
                "eigen_version": version,
                "eigen_age_s": age,
            }
        return {
            "eigen_fitted":    self.is_ready,
            "symbols_tracked": len(self._lob_ema),
            "symbols_seen":    len(self._symbols_seen),
            "symbols_latent":  len(self._lob_latent),
            "buckets":         bucket_info,
            "regime":          self.get_regime_state(),
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _normalize_lob(self, bids: List, asks: List, mid: float) -> Optional[np.ndarray]:
        K = self._K
        bid_p = [float(b[0]) for b in bids[:K]]
        bid_v = [float(b[1]) for b in bids[:K]]
        ask_p = [float(a[0]) for a in asks[:K]]
        ask_v = [float(a[1]) for a in asks[:K]]

        while len(bid_p) < K:
            bid_p.append(bid_p[-1] if bid_p else mid * 0.9999)
            bid_v.append(0.0)
        while len(ask_p) < K:
            ask_p.append(ask_p[-1] if ask_p else mid * 1.0001)
            ask_v.append(0.0)

        total_depth = sum(bid_v) + sum(ask_v)
        if total_depth < 1e-12:
            return None

        bp = [(p - mid) / mid for p in bid_p]
        ap = [(p - mid) / mid for p in ask_p]
        bv = [v / total_depth for v in bid_v]
        av = [v / total_depth for v in ask_v]
        return np.array(bp + bv + ap + av, dtype=np.float64)

    def _get_bucket(self, symbol: str, now: float) -> str:
        """
        [必改4] 带迟滞保护的桶分配。

        流程：
          1. 更新事件频率 EWMA → 计算原始桶 raw_bucket
          2. 若 raw_bucket == 当前桶 → 维持不变，重置候选计时器
          3. 若 raw_bucket != 当前桶 → 进入候选等待
          4. 候选持续 BUCKET_HYSTERESIS_S 秒后才正式切换

        防止：事件率短时波动（如 1 笔大单突发）导致坐标系跳变。
        """
        # 更新事件频率 EWMA
        prev_event = self._sym_last_event.get(symbol)
        if prev_event is not None:
            dt_event  = max(1e-3, now - prev_event)
            inst_rate = 1.0 / dt_event
            old_rate  = self._sym_event_rate.get(symbol, inst_rate)
            decay     = math.exp(-dt_event / (10.0 / math.log(2.0)))
            self._sym_event_rate[symbol] = decay * old_rate + (1.0 - decay) * inst_rate
        else:
            self._sym_event_rate[symbol] = 1.0
        self._sym_last_event[symbol] = now

        rate = self._sym_event_rate[symbol]
        raw_bucket = LIQUIDITY_BUCKETS[-1][0]
        for name, lo, hi in LIQUIDITY_BUCKETS:
            if lo <= rate < hi:
                raw_bucket = name
                break

        # 首次分配：直接生效
        if symbol not in self._sym_bucket:
            self._sym_bucket[symbol]           = raw_bucket
            self._sym_bucket_candidate[symbol] = raw_bucket
            self._sym_candidate_since[symbol]  = now
            return raw_bucket

        current = self._sym_bucket[symbol]

        if raw_bucket == current:
            # 无切换意图，重置候选
            self._sym_bucket_candidate[symbol] = raw_bucket
            self._sym_candidate_since[symbol]  = now
            return current

        # 有切换意图：记录候选，判断是否满足迟滞时间
        candidate = self._sym_bucket_candidate.get(symbol)
        if candidate != raw_bucket:
            # 候选本身变了，重新计时
            self._sym_bucket_candidate[symbol] = raw_bucket
            self._sym_candidate_since[symbol]  = now
            return current

        # 候选稳定，检查持续时长
        if now - self._sym_candidate_since[symbol] >= BUCKET_HYSTERESIS_S:
            self._sym_bucket[symbol]           = raw_bucket
            self._sym_bucket_candidate[symbol] = raw_bucket
            self._sym_candidate_since[symbol]  = now
            logger.debug(f"[LOBManifold] {symbol} 桶切换: {current} → {raw_bucket}")
            return raw_bucket

        return current

    def _extract_eigenvectors(self, bucket: _BucketCovState):
        """
        [必改2/3/5] 从 EWMA 协方差提取特征向量。

        必改2：variance_ratio = λ₁ / trace(cov_work)，不再是前 N 个特征值之和。
        必改3：np.linalg.eigh 替代幂迭代 + deflation，避免误差传导。
        必改5：写入 eigen_ts 和 eigen_version，支持时序一致性监控。
        """
        try:
            with bucket.lock:
                cov_snap       = bucket.cov.copy() if bucket.cov is not None else None
                prev_pc1_vec   = bucket.pc1_vec
                prev_components = bucket.components

            if cov_snap is None:
                return

            # Shrinkage：向对角线收缩
            diag_mat = np.diag(np.diag(cov_snap))
            cov_work = (1.0 - self._eta) * cov_snap + self._eta * diag_mat

            # [必改3] np.linalg.eigh：实对称矩阵，稳定精确，无 deflation 误差传导
            # eigh 返回升序特征值，需翻转为降序
            eigvals, eigvecs = np.linalg.eigh(cov_work)
            idx     = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]   # 每列是一个特征向量

            # 取前 n_comp+1 个（PC1 + lob_z1/z2/z3 的基底）
            n_total    = self._n_comp + 1
            comps      = [eigvecs[:, k].copy() for k in range(n_total)]
            eigenvals_list = [float(max(eigvals[k], 1e-10)) for k in range(n_total)]

            # [必改5] 符号锚定：与上次特征向量做点积，负数时翻转
            if prev_pc1_vec is not None:
                if float(comps[0] @ prev_pc1_vec) < 0:
                    comps[0] = -comps[0]
            if prev_components is not None:
                for k in range(1, n_total):
                    if float(comps[k] @ prev_components[k - 1]) < 0:
                        comps[k] = -comps[k]

            pc1_vec      = comps[0]
            pc1_eigenval = eigenvals_list[0]

            components_mat     = np.array(comps[1:])           # (n_comp, D)
            component_eigenvals = np.array(eigenvals_list[1:]) # (n_comp,)

            # [必改2] variance_ratio 分母 = trace(cov_work)，等于所有 D 个特征值之和
            total_var      = float(np.trace(cov_work))
            variance_ratio = pc1_eigenval / (total_var + 1e-10)

            with bucket.lock:
                bucket.pc1_vec             = pc1_vec
                bucket.pc1_eigenval        = pc1_eigenval
                bucket.components          = components_mat
                bucket.component_eigenvals = component_eigenvals
                bucket.variance_ratio      = variance_ratio
                bucket.fitted              = True
                bucket.eigen_ts            = time.time()   # [必改5]
                bucket.eigen_version      += 1             # [必改5]

            logger.debug(
                f"[LOBManifold] 特征向量更新 v{bucket.eigen_version} "
                f"PC1_ratio={variance_ratio:.3f} "
                f"top4_λ={[f'{eigvals[k]:.4f}' for k in range(min(4, len(eigvals)))]}"
            )

        except Exception as e:
            logger.error(f"[LOBManifold] 特征向量提取失败: {e}")
        finally:
            bucket.eigen_in_progress = False

    def _project_and_smooth(
        self, symbol: str, x: np.ndarray, bucket: _BucketCovState
    ):
        """
        白化投影 + EMA 平滑。

        z̃_k = (PC_{k+1}ᵀ · x) / √λ_{k+1}

        Var(z̃_k) ≈ 1，各维度强度可比（马氏距离分解）。
        PC1 白化分数单独存入 _pc1_scores，仅用于 shock_regime 检测。
        """
        with bucket.lock:
            if not bucket.fitted:
                return
            components     = bucket.components
            comp_eigenvals = bucket.component_eigenvals
            pc1_vec        = bucket.pc1_vec
            pc1_eigenval   = bucket.pc1_eigenval

        # 白化投影
        z_raw     = components @ x
        z_new     = z_raw / (np.sqrt(comp_eigenvals) + 1e-10)
        pc1_score = float(pc1_vec @ x) / (math.sqrt(pc1_eigenval) + 1e-10)

        # EMA 平滑
        if symbol in self._lob_latent:
            z = self._z_ema * z_new + (1.0 - self._z_ema) * self._lob_latent[symbol]
        else:
            z = z_new.copy()
        self._lob_latent[symbol] = z

        if symbol in self._pc1_scores:
            self._pc1_scores[symbol] = (
                self._z_ema * pc1_score + (1.0 - self._z_ema) * self._pc1_scores[symbol]
            )
        else:
            self._pc1_scores[symbol] = pc1_score

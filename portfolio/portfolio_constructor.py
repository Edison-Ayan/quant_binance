"""
Portfolio Constructor - 目标组合构建器

核心范式升级：
    旧版：有信号就开，仓位固定 100 USDT
    新版：根据 unified_alpha 强度 + 风险预算 → 生成连续目标仓位

    target_position[symbol] = f(alpha_strength, risk_budget, correlation, cost)

构建步骤：
    1. 过滤不可交易标的（spread 过宽 / 无信号 / 被 ShockDetector 暂停）
    2. 按 |unified_alpha| 排序候选
    3. 相关性去重：alpha 相似且 ret_1m 高度相关的保留最强
    4. Beta-中性约束：多头总名义价值 ≈ 空头总名义价值（±max_net_exposure）
    5. 按 alpha 强度在风险预算内按比例分配仓位
    6. 返回 target_portfolio = {symbol: signed_notional_usdt}
       正值 = 做多，负值 = 做空，0 = 平仓（不在 target 中的当前持仓）

仓位大小计算：
    base_size = params["trade_size_usdt"]
    alpha_scale = |unified| / mean(|unified|) in candidates → [min_scale, max_scale]
    target_notional = base_size × alpha_scale × portfolio_weight

portfolio_weight：
    基于历史 Sharpe proxy（该品种近期胜率 × 盈亏比）
    范围 [0.5, 1.5]
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TargetPosition:
    symbol:        str
    side:          str     # "LONG" / "SHORT"
    target_usdt:   float   # 目标名义价值（正值）
    unified_alpha: float   # 对应的 unified alpha
    reason:        str     # 进入原因（"new_long", "new_short", "keep_long", ...）


class PortfolioConstructor:
    """
    目标组合构建器。

    接口：
        build(fused_alphas, features, current_longs, current_shorts,
              shock_detector, symbol_stats, params) → TargetPortfolio

    返回：
        TargetPortfolio.longs  : {symbol: TargetPosition}
        TargetPortfolio.shorts : {symbol: TargetPosition}
        TargetPortfolio.to_close_long  : [symbol]  需平仓的多头
        TargetPortfolio.to_close_short : [symbol]  需平仓的空头
    """

    def __init__(
        self,
        max_long_positions:  int   = 3,
        max_short_positions: int   = 3,
        max_net_exposure:    float = 300.0,
        base_size_usdt:      float = 100.0,
        min_alpha_scale:     float = 0.7,
        max_alpha_scale:     float = 1.5,
        max_corr_threshold:  float = 0.85,
        corr_history_len:    int   = 30,
        port_weight_scale:   float = 0.4,
        port_min_trades:     int   = 5,
    ):
        self.max_long        = max_long_positions
        self.max_short       = max_short_positions
        self.max_net         = max_net_exposure
        self.base_size       = base_size_usdt
        self.min_scale       = min_alpha_scale
        self.max_scale       = max_alpha_scale
        self.max_corr        = max_corr_threshold
        self.corr_len        = corr_history_len
        self.port_w_scale    = port_weight_scale
        self.port_min_trades = port_min_trades

    def build(
        self,
        fused_alphas:    dict,           # {symbol: FusedAlpha}
        features:        dict,           # {symbol: SymbolFeatures}
        current_longs:   dict,           # {symbol: position_dict}
        current_shorts:  dict,
        shock_detector,                  # ShockDetector instance
        symbol_stats:    dict,           # {symbol: deque of pnl values}
        ret1m_history:   dict,           # {symbol: deque of ret_1m values}
        max_spread_bps:  float = 20.0,
    ) -> "TargetPortfolio":
        """
        构建目标组合。返回 TargetPortfolio。
        """

        # ── 1. 过滤候选 ───────────────────────────────────────────────────
        long_candidates  = []   # [(symbol, unified_alpha)]
        short_candidates = []

        for sym, fa in fused_alphas.items():
            feat = features.get(sym)
            if feat is None:
                continue
            # 过滤：盘口太宽
            if feat.spread_bps > max_spread_bps:
                continue
            # 过滤：插针暂停
            if shock_detector and shock_detector.is_paused(sym):
                continue

            if fa.is_long_candidate:
                long_candidates.append((sym, fa.unified))
            elif fa.is_short_candidate:
                short_candidates.append((sym, fa.unified))

        # 按 alpha 强度排序
        long_candidates  = sorted(long_candidates,  key=lambda x: x[1],  reverse=True)
        short_candidates = sorted(short_candidates, key=lambda x: x[1])

        # ── 2. 相关性去重 ─────────────────────────────────────────────────
        long_candidates  = self._dedup_by_corr(long_candidates,  ret1m_history)
        short_candidates = self._dedup_by_corr(short_candidates, ret1m_history)

        # 互斥：同品种不能同时进入多空两侧
        long_syms  = {s for s, _ in long_candidates}
        short_candidates = [(s, a) for s, a in short_candidates if s not in long_syms]

        # ── 3. Beta 中性约束 ──────────────────────────────────────────────
        long_candidates, short_candidates = self._apply_net_exposure(
            long_candidates, short_candidates
        )

        # ── 4. 截断到最大持仓数 ──────────────────────────────────────────
        long_candidates  = long_candidates[:self.max_long]
        short_candidates = short_candidates[:self.max_short]

        # ── 5. 计算目标仓位大小 ───────────────────────────────────────────
        target_longs  = self._assign_sizes(long_candidates,  "LONG",  symbol_stats)
        target_shorts = self._assign_sizes(short_candidates, "SHORT", symbol_stats)

        # ── 6. 确定需要平仓的品种 ────────────────────────────────────────
        target_long_syms  = set(target_longs.keys())
        target_short_syms = set(target_shorts.keys())

        to_close_long  = [s for s in current_longs  if s not in target_long_syms]
        to_close_short = [s for s in current_shorts if s not in target_short_syms]

        # 保留已有仓位（不在 fused_alphas 中的，视为 alpha 消失 → 平仓）
        for sym in list(current_longs.keys()):
            if sym in target_long_syms:
                target_longs[sym].reason = "keep_long"

        for sym in list(current_shorts.keys()):
            if sym in target_short_syms:
                target_shorts[sym].reason = "keep_short"

        return TargetPortfolio(
            longs          = target_longs,
            shorts         = target_shorts,
            to_close_long  = to_close_long,
            to_close_short = to_close_short,
        )

    # ─── 内部方法 ─────────────────────────────────────────────────────────────

    def _dedup_by_corr(
        self,
        candidates: List[Tuple[str, float]],
        ret1m_history: dict,
    ) -> List[Tuple[str, float]]:
        """
        相关性去重：alpha 最强的优先保留，后续候选与已选中任一品种相关性超阈值则丢弃。
        """
        selected = []
        for sym, alpha in candidates:
            dominated = False
            for sel_sym, _ in selected:
                corr = self._calc_corr(sym, sel_sym, ret1m_history)
                if corr is not None and corr > self.max_corr:
                    dominated = True
                    break
            if not dominated:
                selected.append((sym, alpha))
        return selected

    def _calc_corr(self, sym_a: str, sym_b: str, ret1m_history: dict) -> Optional[float]:
        """计算两个品种 ret_1m 序列的皮尔逊相关系数"""
        hist_a = ret1m_history.get(sym_a)
        hist_b = ret1m_history.get(sym_b)
        if not hist_a or not hist_b or len(hist_a) < 5 or len(hist_b) < 5:
            return None
        n = min(len(hist_a), len(hist_b), self.corr_len)
        a_vals = list(hist_a)[-n:]
        b_vals = list(hist_b)[-n:]
        return _pearson_corr(a_vals, b_vals)

    def _apply_net_exposure(
        self,
        longs:  List[Tuple[str, float]],
        shorts: List[Tuple[str, float]],
    ) -> Tuple[List, List]:
        """
        简单 Beta 中性约束：
        如果预期多头总名义 - 空头总名义 > max_net_exposure，
        则裁剪多头列表末尾的候选（最弱的先删）。
        反之亦然。
        """
        long_count  = min(len(longs),  self.max_long)
        short_count = min(len(shorts), self.max_short)

        est_long_notional  = long_count  * self.base_size * self.max_scale
        est_short_notional = short_count * self.base_size * self.max_scale

        net = est_long_notional - est_short_notional
        while abs(net) > self.max_net and long_count > 0 and short_count > 0:
            if net > 0:
                long_count -= 1
            else:
                short_count -= 1
            est_long_notional  = long_count  * self.base_size * self.max_scale
            est_short_notional = short_count * self.base_size * self.max_scale
            net = est_long_notional - est_short_notional

        return longs[:long_count], shorts[:short_count]

    def _assign_sizes(
        self,
        candidates:   List[Tuple[str, float]],
        side:         str,
        symbol_stats: dict,
    ) -> Dict[str, TargetPosition]:
        """
        按 alpha 强度和历史 Sharpe proxy 分配连续仓位大小。
        """
        if not candidates:
            return {}

        alphas = [abs(a) for _, a in candidates]
        mean_alpha = sum(alphas) / len(alphas) if alphas else 1.0

        result = {}
        for sym, alpha_val in candidates:
            # alpha 比例因子
            rel  = abs(alpha_val) / (mean_alpha + 1e-10)
            size_scale = self.min_scale + (self.max_scale - self.min_scale) * min(rel, 2.0) / 2.0
            size_scale = max(self.min_scale, min(self.max_scale, size_scale))

            # 历史 Sharpe proxy 权重
            port_weight = self._calc_port_weight(sym, symbol_stats)
            target_usdt = self.base_size * size_scale * port_weight

            result[sym] = TargetPosition(
                symbol        = sym,
                side          = side,
                target_usdt   = round(target_usdt, 2),
                unified_alpha = alpha_val,
                reason        = "new_long" if side == "LONG" else "new_short",
            )
        return result

    def _calc_port_weight(self, sym: str, symbol_stats: dict) -> float:
        """
        基于历史交易 pnl 序列估计 Sharpe proxy，转换为 [0.5, 1.5] 的权重。
        数据不足时返回 1.0。
        """
        stats = symbol_stats.get(sym)
        if not stats or len(stats) < self.port_min_trades:
            return 1.0

        pnl_list = list(stats)
        n        = len(pnl_list)
        mean_pnl = sum(pnl_list) / n
        std_pnl  = math.sqrt(max(sum((p - mean_pnl) ** 2 for p in pnl_list) / n, 1e-10))
        sharpe   = mean_pnl / std_pnl

        # 把 Sharpe 映射到 [0.5, 1.5]
        weight = 1.0 + self.port_w_scale * math.tanh(sharpe)
        return max(0.5, min(1.5, weight))


@dataclass
class TargetPortfolio:
    longs:          Dict[str, TargetPosition]
    shorts:         Dict[str, TargetPosition]
    to_close_long:  List[str]
    to_close_short: List[str]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _pearson_corr(a: list, b: list) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[-n:], b[-n:]
    mu_a = sum(a) / n
    mu_b = sum(b) / n
    num  = sum((a[i] - mu_a) * (b[i] - mu_b) for i in range(n))
    da   = math.sqrt(max(sum((v - mu_a) ** 2 for v in a), 1e-20))
    db   = math.sqrt(max(sum((v - mu_b) ** 2 for v in b), 1e-20))
    return num / (da * db)

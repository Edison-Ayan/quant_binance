"""
Cost Model - 交易前成本估计

职责：
    在开仓前估算此笔交易的全成本（手续费 + 点差 + 市场冲击），
    与预期 alpha 收益对比，判断 edge 是否足够覆盖成本。

    这是"成本驱动决策"的核心：
    很多看起来有信号的交易，实际上 alpha < cost，是亏钱的。

成本组成：
    1. 手续费（双边）  : notional × fee_rate × 2
    2. 半点差成本     : notional × spread_bps / 20000  （买入即亏半个价差）
    3. 市场冲击估计   : (notional / depth_usdt) × notional × impact_factor
                        → 简化 Almgren-Chriss 线性冲击模型
                        → 当 notional / depth_usdt < 1% 时冲击可忽略

预期 alpha 收益估计：
    expected_return = |unified_alpha| × avg_holding_seconds / time_to_halflife × base_ret_per_sigma
    （保守估计，使用历史均值）

    实际实现中我们用更保守的方式：
    expected_gross = |unified_alpha| × expected_ret_per_unit
    where expected_ret_per_unit ≈ 0.002 (历史均值，每单位 Z-score 对应约 0.2% 毛收益)

is_viable 判断：
    expected_gross - total_cost > min_edge_usdt

接口：
    cost_model.estimate(symbol, notional, features, unified_alpha) → CostEstimate
    cost_model.is_viable(estimate) → bool
"""

from dataclasses import dataclass
from typing import Optional


# ── 默认参数 ──────────────────────────────────────────────────────────────────

DEFAULT_FEE_RATE         = 0.0004    # Binance taker fee（0.04%）
DEFAULT_IMPACT_FACTOR    = 0.001     # 市场冲击系数（保守估计）
DEFAULT_MIN_EDGE_MULT    = 1.2       # 预期 alpha 收益必须是成本的 N 倍
EXPECTED_RET_PER_UNIT    = 0.0015   # 每单位 unified_alpha 对应预期毛收益（0.15%）
MIN_EXPECTED_GROSS_USDT  = 0.012    # 预期毛收益绝对门槛（USDT），约等于双边手续费


@dataclass
class CostEstimate:
    symbol:           str
    notional:         float

    # 成本分解
    fee_cost:         float = 0.0    # 双边手续费
    spread_cost:      float = 0.0    # 半点差成本
    impact_cost:      float = 0.0    # 市场冲击
    total_cost:       float = 0.0    # 总成本（USDT）
    cost_bps:         float = 0.0    # 总成本（基点）

    # 预期收益
    expected_gross:   float = 0.0    # 预期毛收益（USDT）
    net_edge:         float = 0.0    # 净 edge = expected_gross - total_cost

    # 判断
    is_viable:        bool  = False
    reject_reason:    str   = ""


class CostModel:
    """
    交易前成本估计器。

    接口：
        estimate(symbol, notional, unified_alpha, spread_bps, depth_usdt) → CostEstimate
    """

    def __init__(
        self,
        fee_rate:          float = DEFAULT_FEE_RATE,
        impact_factor:     float = DEFAULT_IMPACT_FACTOR,
        min_edge_multiple: float = DEFAULT_MIN_EDGE_MULT,
        ret_per_unit:      float = EXPECTED_RET_PER_UNIT,
    ):
        self.fee_rate    = fee_rate
        self.impact      = impact_factor
        self.min_edge    = min_edge_multiple
        self.ret_per_unit = ret_per_unit

    def estimate(
        self,
        symbol:        str,
        notional:      float,
        unified_alpha: float,
        spread_bps:    float = 5.0,
        depth_usdt:    float = 100_000.0,
    ) -> CostEstimate:
        """
        估算一笔交易的完整成本并判断是否值得做。

        参数：
            notional       : 交易名义价值（USDT）
            unified_alpha  : 来自 AlphaFusionEngine 的融合 alpha 分数
            spread_bps     : 当前买卖点差（基点），来自 features.spread_bps
            depth_usdt     : 盘口最优一档的 USDT 深度（bid_usdt + ask_usdt）
        """
        if notional <= 0:
            return CostEstimate(symbol=symbol, notional=notional,
                                reject_reason="zero_notional")

        # ── 手续费 ────────────────────────────────────────────────────────
        fee_cost    = notional * self.fee_rate * 2     # 开 + 平 双边

        # ── 半点差成本 ────────────────────────────────────────────────────
        spread_cost = notional * spread_bps / 20_000.0  # bps → fraction，买入即亏半个价差

        # ── 市场冲击 ──────────────────────────────────────────────────────
        # 简化线性冲击：sigma_impact = impact_factor × notional / depth
        participation = notional / max(depth_usdt, 1.0)
        impact_cost   = self.impact * participation * notional

        # ── 总成本 ────────────────────────────────────────────────────────
        total_cost = fee_cost + spread_cost + impact_cost
        cost_bps   = total_cost / notional * 10_000

        # ── 预期毛收益 ────────────────────────────────────────────────────
        # 保守估计：每单位 unified_alpha 对应 ret_per_unit 的毛收益
        expected_gross = abs(unified_alpha) * self.ret_per_unit * notional

        # ── 净 edge ───────────────────────────────────────────────────────
        net_edge   = expected_gross - total_cost
        is_viable  = (expected_gross >= self.min_edge * total_cost
                      and expected_gross >= MIN_EXPECTED_GROSS_USDT)

        reject_reason = ""
        if not is_viable:
            if expected_gross < MIN_EXPECTED_GROSS_USDT:
                reject_reason = (
                    f"below_min_gross: expected={expected_gross:.4f} < {MIN_EXPECTED_GROSS_USDT}"
                )
            else:
                reject_reason = (
                    f"edge_insufficient: expected={expected_gross:.4f} "
                    f"cost={total_cost:.4f} "
                    f"ratio={expected_gross/(total_cost+1e-10):.2f}x < {self.min_edge}x"
                )

        return CostEstimate(
            symbol         = symbol,
            notional       = notional,
            fee_cost       = round(fee_cost,     4),
            spread_cost    = round(spread_cost,  4),
            impact_cost    = round(impact_cost,  4),
            total_cost     = round(total_cost,   4),
            cost_bps       = round(cost_bps,     2),
            expected_gross = round(expected_gross, 4),
            net_edge       = round(net_edge,     4),
            is_viable      = is_viable,
            reject_reason  = reject_reason,
        )

    def estimate_from_features(
        self,
        symbol:        str,
        notional:      float,
        unified_alpha: float,
        features,                  # SymbolFeatures
    ) -> CostEstimate:
        """
        从 SymbolFeatures 自动读取 spread_bps 和 depth_usdt（简化深度估计）。
        """
        spread_bps = getattr(features, "spread_bps", 5.0)

        # 优先使用 bookTicker 实时双边深度（bid×bid_qty + ask×ask_qty）
        # 回退到 last_price×10 粗估（仅在数据未就绪时）
        best_depth = getattr(features, "best_depth_usdt", 0.0)
        if best_depth > 1.0:
            depth_usdt = best_depth
        else:
            last_price = getattr(features, "last_price", 1.0)
            depth_usdt = max(last_price * 10.0, 200.0)   # 至少 200 USDT 下限

        return self.estimate(
            symbol        = symbol,
            notional      = notional,
            unified_alpha = unified_alpha,
            spread_bps    = spread_bps,
            depth_usdt    = depth_usdt,
        )

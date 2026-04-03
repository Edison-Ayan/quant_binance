# Institutional Crypto Quant — Alpha Operating System

生产级加密货币量化交易系统，基于 Binance Futures，实现**全市场横截面 Alpha 工厂**与**LOB 微观结构实时引擎**的深度融合。系统不依赖单一信号触发，而是以**连续 Alpha 流**驱动持仓生命周期管理，实现自适应、自校正的智能交易决策。

---

## 目录

1. [策略定位与设计哲学](#1-策略定位与设计哲学)
2. [系统总体架构](#2-系统总体架构)
3. [完整策略流程（Step by Step）](#3-完整策略流程step-by-step)
   - 3.1 [数据采集层](#31-数据采集层)
   - 3.2 [特征引擎层](#32-特征引擎层)
   - 3.3 [LOB 流形引擎](#33-lob-流形引擎)
   - 3.4 [慢层评分引擎（横截面 Alpha）](#34-慢层评分引擎横截面-alpha)
   - 3.5 [市场状态引擎](#35-市场状态引擎)
   - 3.6 [快层时序引擎（LOB Timing）](#36-快层时序引擎lob-timing)
   - 3.7 [Alpha 融合引擎](#37-alpha-融合引擎)
   - 3.8 [持仓生命周期状态机](#38-持仓生命周期状态机)
   - 3.9 [组合构建引擎](#39-组合构建引擎)
   - 3.10 [交易前成本模型](#310-交易前成本模型)
   - 3.11 [风控体系（三重门禁）](#311-风控体系三重门禁)
   - 3.12 [执行层与订单管理](#312-执行层与订单管理)
   - 3.13 [持仓管理与浮盈更新](#313-持仓管理与浮盈更新)
   - 3.14 [监控引擎与报告引擎](#314-监控引擎与报告引擎)
   - 3.15 [异步数据库存储](#315-异步数据库存储)
4. [Alpha 融合公式推导](#4-alpha-融合公式推导)
5. [因子工程详解](#5-因子工程详解)
6. [LOB PCA 微观结构提取](#6-lob-pca-微观结构提取)
7. [持仓生命周期状态机详解](#7-持仓生命周期状态机详解)
8. [组合构建与仓位定价](#8-组合构建与仓位定价)
9. [风控设计](#9-风控设计)
10. [事件驱动架构](#10-事件驱动架构)
11. [模块目录结构](#11-模块目录结构)
12. [数据库 Schema](#12-数据库-schema)
13. [配置与部署](#13-配置与部署)
14. [运行方式](#14-运行方式)

---

## 1. 策略定位与设计哲学

### 1.1 核心理念：连续 Alpha 流，而非离散信号

大多数传统策略是"信号触发型"：当某个指标穿越阈值时，触发一次买入或卖出动作，之后对持仓不闻不问，等待止盈止损触发平仓。

本系统的设计哲学完全不同——**持仓是被 Alpha 强度所驱动的连续过程，而不是被信号触发的离散事件。**

- 每 60 秒，系统对全市场 270+ 个品种进行一次横截面评分
- 每一笔持仓都有一个实时更新的 "unified alpha" 分数
- 该分数融合了慢层排名、快层微观结构、市场状态、拥挤度折扣
- 当一笔持仓的 alpha 分数从峰值衰减并出现反转迹象时，系统主动平仓
- 平仓不是因为价格到了止盈止损线，而是因为 **alpha 信号本身失效了**

### 1.2 横截面市场中性

不预测 BTC 或任何单一品种的绝对涨跌，只做「相对强弱」排序：
- 在所有候选币中，找出**相对最强、尚未充分反映的品种**做多
- 同时找出**相对最弱、尚未充分反映的品种**做空
- 多空组合构成市场中性，大幅降低对 BTC 系统性行情的暴露

### 1.3 双层 Alpha 架构

| 层级 | 名称 | 频率 | 作用 |
|------|------|------|------|
| 慢层 | ScoringEngine | 每 60s | 横截面多因子评分，决定做多/做空候选池 |
| 快层 | LOBTimingEngine | 逐 Tick | LOB 微观结构时序判断，决定是否立即进场 |

慢层决定「做什么」，快层决定「什么时候做」。两者在 AlphaFusionEngine 中合并为统一的 unified alpha 分数。

### 1.4 系统规格

| 参数 | 规格 |
|------|------|
| 覆盖品种 | 270+ USDT 永续合约（动态过滤） |
| 慢层信号频率 | 每 60 秒一轮排序 |
| 快层信号频率 | 逐 Tick（aggTrade + bookTicker） |
| 持仓管理机制 | Alpha 生命周期状态机 |
| 最大多头持仓 | 3 个（可配置） |
| 最大空头持仓 | 3 个（可配置） |
| 杠杆 | 10x（默认） |
| 执行模式 | Paper Trading / Testnet / Live |

---

## 2. 系统总体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据采集层                                         │
│  MultiSymbolFeed           RestFetcher            UniverseFilter            │
│  (aggTrade + bookTicker    (资金费率 + OI          (交易池过滤                │
│   + depth5 WebSocket)       REST 60s 轮询)          270+ 品种)               │
└───────────────┬───────────────────────┬────────────────────────────────────┘
                │ Tick / OrderBook 事件  │ 资金费率 / OI 直接调用
                ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           特征引擎层 (FeatureEngine)                          │
│  volume_zscore  ofi  ret_1m  ret_5m  spread_bps  depth_imbalance           │
│  funding_rate   oi_change_pct   lob_pc1   lob_z1/z2/z3                     │
└───────────┬──────────────────────────────────────────────────────┬──────────┘
            │ 每 60s 触发                                           │ 逐 Tick
            ▼                                                       ▼
┌───────────────────────┐                            ┌─────────────────────────┐
│  慢层：ScoringEngine  │                            │ LOBManifoldEngine       │
│  横截面多因子评分      │                            │ LOB PCA 提取            │
│  (Z-score 标准化)     │                            │ lob_pc1 / z1 / z2 / z3 │
└───────────┬───────────┘                            └─────────────┬───────────┘
            │ slow_score[]                                         │ LOB 特征
            ▼                                                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  MarketStateEngine                                                         │
│  regime(TRENDING/MEAN_REVERTING/VOLATILE/QUIET)  tradability  dispersion  │
│  crowding_z  is_tradeable                                                  │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │ 市场状态
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  LOBTimingEngine  (快层)                                                    │
│  timing_score = w×microprice + w×OFI + w×lob_pc1 + w×z1 + w×z2 + w×z3   │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │ timing_score
                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  AlphaFusionEngine                                                          │
│  unified = slow × regime_mult × tradability_mult × (1+fast_boost) × crowd  │
└──────────┬─────────────────────────────────────────────────────────────────┘
           │ FusedAlpha{symbol: unified}
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  AlphaLifecycleTracker          PortfolioConstructor                      │
│  BUILD→EXPANSION→DECAY          相关性去重 + Beta 中性                    │
│  →REVERSAL（主动平仓）           + alpha 强度仓位定价                     │
└──────────┬─────────────────────────────────────────────────────────────┬─┘
           │ 持仓状态机控制                                              │ 目标组合
           ▼                                                             ▼
┌───────────────────────┐                                ┌──────────────────────┐
│  CostModel            │                                │  RiskManager         │
│  fee+spread+impact    │                                │  仓位/频率/亏损三重门 │
│  预交易成本过滤        │                                │                      │
└───────────┬───────────┘                                └──────────┬───────────┘
            │ is_viable=True                                        │ check=True
            └─────────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  执行引擎              │
                          │  PaperEngine /         │
                          │  TestnetEngine /       │
                          │  BinanceGateway        │
                          └───────────┬────────────┘
                                      │ FILL 事件
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  PositionManager  Database  Monitor │
                    │  持仓更新  SQLite 异步写入  状态打印  │
                    └─────────────────────────────────────┘
```

---

## 3. 完整策略流程（Step by Step）

### 3.1 数据采集层

#### UniverseFilter — 交易池动态过滤

启动时调用 Binance REST API `/fapi/v1/ticker/24hr`，对全市场 USDT 永续合约进行过滤：

- **成交量门槛**：24h USDT 成交量 ≥ 500 万 USDT
- **排除异常品种**：过滤掉 BUSDUSDT、合规限制品种
- **最大池子规模**：最多选取前 500 个品种（按成交量降序）
- **输出**：`symbols[]` — 本次 session 参与评分和交易的完整品种列表

#### MultiSymbolFeed — 多币 WebSocket

为交易池中每个品种同时维护三条 WebSocket 流：

| 流类型 | Binance 端点 | 数据内容 | 用途 |
|--------|-------------|---------|------|
| aggTrade | `{symbol}@aggTrade` | 逐笔成交（价格、数量、方向） | 成交量因子、动量因子 |
| bookTicker | `{symbol}@bookTicker` | 最优买一卖一实时快照 | OFI、价差、深度不平衡 |
| depth | `{symbol}@depth5@100ms` | 买卖各5档订单簿 | LOB 流形特征 |

所有 WebSocket 回调**绕过事件队列直接调用 FeatureEngine**，消除一次线程切换的延迟。

#### RestFetcher — 衍生品数据轮询

每 60 秒通过 REST API 轮询：
- `/fapi/v1/premiumIndex` → 资金费率（funding_rate）
- `/fapi/v1/openInterest` → 持仓量（OI）

资金费率和 OI 变化率直接写入各品种的 SymbolFeatures，供评分引擎使用。

---

### 3.2 特征引擎层

**文件**：`alpha_factory/feature_engine.py`

每个品种维护一个 `SymbolState` 对象，内含滑动窗口 deque，用于 O(1) 增量更新。所有特征计算均为增量式，不做全量重算。

#### SymbolFeatures 数据结构

```python
@dataclass
class SymbolFeatures:
    # 流量因子
    volume_zscore:    float  # 成交量 Z-score（含方向信息）
    ofi:              float  # Order Flow Imbalance（Cont et al. 2014）

    # 价格动量因子
    ret_1m:           float  # 1 分钟收益率
    ret_5m:           float  # 5 分钟收益率

    # 流动性因子
    spread_bps:       float  # 买卖价差（基点）
    depth_imbalance:  float  # 买卖深度不平衡度

    # 衍生品因子
    funding_rate:     float  # 资金费率
    oi_change_pct:    float  # OI 变化率

    # LOB 微观结构因子（来自 LOBManifoldEngine）
    lob_pc1:          float  # 流形第一主成分得分（冲击状态指标）
    lob_z1:           float  # 第二主成分白化投影（PC2）
    lob_z2:           float  # 第三主成分白化投影（PC3）
    lob_z3:           float  # 第四主成分白化投影（PC4）
```

#### 特征计算逻辑

**volume_zscore（成交量 Z-score）**
```
每笔 aggTrade 到来时：
  signed_vol = price × qty × (+1 if is_buyer_maker=False else -1)
  # is_buyer_maker=True：主动卖出；False：主动买入
  将 signed_vol 压入 deque（窗口 500 条）
  volume_zscore = (signed_vol - mean(deque)) / std(deque)
```

正值：买方主动成交旺盛（看涨信号）
负值：卖方主动成交旺盛（看跌信号）

**OFI（Order Flow Imbalance）**
```
每次 bookTicker 推送：
  bid_change = (bid_qty if bid_p ≥ prev_bid else 0) - (prev_bid_qty if bid_p ≤ prev_bid else 0)
  ask_change = (prev_ask_qty if ask_p ≥ prev_ask else 0) - (ask_qty if ask_p ≤ prev_ask else 0)
  raw_ofi = bid_change - ask_change
  ofi = EMA(raw_ofi, α=0.3)
```
OFI 正值：订单簿买方压力增加（价格上行驱动力）
公式来源：Cont, Kukanov & Stoikov (2014)

**spread_bps（价差）**
```
spread_bps = (ask - bid) / mid_price × 10000
```

**depth_imbalance（深度不平衡）**
```
depth_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
范围 [-1, +1]
```

**ret_1m / ret_5m（价格动量）**
```
每次 aggTrade 更新 price_history deque
ret_1m = (当前价格 - 60秒前价格) / 60秒前价格
ret_5m = (当前价格 - 300秒前价格) / 300秒前价格
```

---

### 3.3 LOB 流形引擎

**文件**：`alpha_factory/lob_manifold_engine.py`（v5 统计严格版）

LOBManifoldEngine 对订单簿进行在线 PCA 降维，提取订单簿几何结构的潜在因子。

#### 输入向量构建

每次收到 depth5 更新（买卖各5档）：

```
LOB 向量 x ∈ R^{20}：
  买方各档：(bid_i - mid) / mid          # 相对价格偏差
             bid_vol_i / total_bid_vol    # 归一化量
  卖方各档：(ask_i - mid) / mid
             ask_vol_i / total_ask_vol
```

#### 流动性分桶（Liquidity Bucketing）

按每品种过去 100 次更新的事件频率，将所有时段分为三桶：
- **high**：高频时段（流动性充沛，价差窄，深度大）
- **mid**：中频时段
- **low**：低频时段（流动性稀缺，价差宽，深度小）

同一桶内的数据共享协方差矩阵，避免不同市场状态的数据混淆 PCA。

#### 分桶 EWMA 标准化

对每桶数据进行 per-feature EWMA 标准化，消除各特征量纲差异：
```
μ_g = EWMA(x_i, α=0.05)
σ_g = sqrt(EWMA((x_i - μ_g)^2, α=0.05))
x̃_i = (x_i - μ_g) / (σ_g + ε)
```

#### 在线 EWMA 协方差与特征分解

```
Σ = EWMA(x̃ · x̃^T, α=0.02)   # α 极小，记忆长达 50 次
加 Shrinkage：Σ_shrunk = (1-δ)Σ + δ·(trace(Σ)/d)·I  （δ=0.1）
使用 np.linalg.eigh 稳定求解特征值 λ₁ ≥ λ₂ ≥ ...
```

#### 白化投影输出

```
z_k = v_k^T · x̃    （在特征向量 v_k 上的投影）
z̃_k = z_k / sqrt(λ_k)  （白化：使各维度方差统一为 1）
z̃_k_smoothed = EMA(z̃_k, α=0.4)
```

最终输出：
- `lob_pc1`：第一主成分白化得分（捕捉订单簿整体结构性变化，高绝对值 = shock）
- `lob_z1/z2/z3`：第二、三、四主成分白化投影（捕捉订单簿细微形态变化）

符号稳定性保证：每次特征向量更新后，检查与历史向量的方向一致性，如内积 < 0 则取反，避免 PCA 符号翻转问题。

---

### 3.4 慢层评分引擎（横截面 Alpha）

**文件**：`alpha_factory/scoring_engine.py`

每 60 秒对全市场品种做一次横截面评分，排出相对强弱。

#### 因子权重（默认）

| 因子 | 权重 | 方向 | 含义 |
|------|------|------|------|
| volume_zscore | 0.30 | +1.0 | 主动买入越强，看涨信号越强 |
| oi_change_pct | 0.25 | +1.0 | 持仓量增加，趋势有资金支撑 |
| ret_1m | 0.20 | +1.0 | 近期动量追踪 |
| ret_5m | 0.15 | +1.0 | 中短期动量确认 |
| funding_rate | 0.10 | -1.0 | 资金费率高→多头拥挤→反转信号 |

#### 横截面 Z-score 标准化流程

```
1. 收集所有品种的各因子原始值
2. 对每个因子 f 做 Winsorization：clip(x, μ ± 3σ) — 去除异常值
3. 横截面 Z-score：z_f(symbol) = (x_f - mean(x_f)) / std(x_f)
4. 加权求和：raw_score = Σ weight_f × direction_f × z_f
5. 再做一次全市场 Z-score 归一化：
   slow_score = (raw_score - mean) / std
```

最终 slow_score 以 0 为中心，正值越大越强，负值越大越弱，可直接比较。

#### IC 自适应权重进化

系统记录每个因子的 IC（实际收益率与因子值的相关系数），通过 EMA 动态更新权重：
```
weight_f(t+1) = weight_f(t) × (1 - α) + IC_f(t) × α  （α = 0.1）
归一化：weight_f = weight_f / Σ weight_f
```

---

### 3.5 市场状态引擎

**文件**：`alpha_factory/market_state_engine.py`

在每次横截面评分前，先判断当前全市场状态，决定是否适合交易及 Alpha 调整系数。

#### Regime 判断逻辑

```
dispersion = cross_section_std(ret_1m)  # 横截面收益率离散度

IF dispersion > 0.8%:
    regime = VOLATILE     # 市场波动剧烈，风险过高
ELIF |BTC_ret_5m| > 1.5%:
    regime = TRENDING     # BTC 强趋势，跟随趋势
ELIF dispersion > 0.3% AND |BTC_ret_5m| < 0.5%:
    regime = MEAN_REVERTING  # 低趋势高分散，适合反转
ELSE:
    regime = QUIET        # 低波动低分散，降低交易频率
```

#### Tradability（可交易性评估）

```
spread_health = exp(-avg_spread_bps / 10)   # 价差越窄越好
activity_score = 活跃品种数 / 总品种数
tradability = 0.6 × spread_health + 0.4 × activity_score
范围 [0, 1]
```

#### Crowding Z-score（拥挤度）

```
crowding_z = (avg_funding_rate - hist_mean) / hist_std
```
高 crowding_z → 多头拥挤 → 对做多 Alpha 施加折扣

#### 是否可交易（is_tradeable）

```python
is_tradeable = (
    tradability >= 0.30           # 流动性最低要求
    AND dispersion >= 0.002       # 有足够 Alpha 机会（0.2%）
    AND regime != VOLATILE        # 排除极端波动状态
)
```

如果 `is_tradeable = False`，系统**跳过本轮所有开仓决策**。

#### Regime 乘数（regime_mult）

| Regime | regime_mult | 含义 |
|--------|-------------|------|
| TRENDING | 1.00 | 趋势行情，Alpha 权重不变 |
| MEAN_REVERTING | 0.90 | 反转行情，轻微降低 Alpha 权重 |
| VOLATILE | 0.60 | 极端波动，显著缩减 Alpha |
| QUIET | 0.70 | 低活跃，适当降低 Alpha |

---

### 3.6 快层时序引擎（LOB Timing）

**文件**：`alpha_factory/lob_timing_engine.py`

快层不预测价格方向（这是慢层的工作），只判断「此刻的微观结构是否利于进场」。

#### Timing Score 公式

```
timing_score = (
    W_MICROPRICE × microprice_norm   # 权重 0.30
  + W_OFI       × soft_clip(ofi)    # 权重 0.25
  + W_LOB_PC1   × soft_clip(lob_pc1)  # 权重 0.15
  + W_LOB_Z1    × soft_clip(lob_z1)   # 权重 0.12
  + W_LOB_Z2    × soft_clip(lob_z2)   # 权重 0.10
  + W_LOB_Z3    × soft_clip(lob_z3)   # 权重 0.08
)
```

**soft_clip**：`tanh(x / 3.0) × 3.0`，防止单因子极端值主导得分。

#### Microprice 计算与标准化

```
Microprice = (bid × ask_qty + ask × bid_qty) / (bid_qty + ask_qty)
microprice_delta = microprice - mid_price  # 微价差偏移

# 滚动 Z-score（窗口 200 次）
microprice_norm = (delta - rolling_mean) / (rolling_std + ε)
```

#### 进场/离场判断

```python
# 做多进场时序确认
should_enter_long = timing_score > ENTRY_THRESHOLD(0.40)

# 做空进场时序确认（时序得分应为负值，说明卖压强）
should_enter_short = timing_score < -ENTRY_THRESHOLD(-0.40)

# 已持多头，考虑时序离场
should_exit_long = timing_score < -EXIT_THRESHOLD(-0.20)

# 已持空头，考虑时序离场
should_exit_short = timing_score > EXIT_THRESHOLD(0.20)
```

冷却机制：同一品种两次时序触发间隔 ≥ 5 秒，避免抖动。

---

### 3.7 Alpha 融合引擎

**文件**：`alpha_factory/alpha_fusion.py`

AlphaFusionEngine 将慢层分数、快层时序、市场状态、拥挤度折扣统一融合为一个可直接驱动仓位决策的 `unified_alpha` 分数。

#### 融合公式（详细推导见第 4 节）

```
unified = slow_score × regime_mult × tradability_mult × (1 + fast_boost) × crowding_disc
```

#### 各项定义

**slow_score**：来自 ScoringEngine 的横截面 Z-score，反映品种相对强弱

**regime_mult**：MarketStateEngine 的市场状态乘数（0.60 ~ 1.00）

**tradability_mult**：
```
tradability_mult = max(0.50, tradability)
```
流动性过差时自动缩减信号强度

**fast_boost**：
```python
# 做多候选（slow_score > 0）
if timing_score > 0:
    fast_boost = +W_FAST_BOOST × timing_score  # 最大 +0.50，顺时序增强
else:
    fast_boost = -W_FAST_DRAG × |timing_score|  # 最大 -0.40，逆时序减弱

# 做空候选（slow_score < 0）反向处理
```

**crowding_disc（拥挤度折扣）**：
```python
if side == "LONG" and crowding_z > 1.5:
    excess = crowding_z - 1.5
    crowding_disc = max(0.50, 1.0 - 0.15 × excess)
else:
    crowding_disc = 1.0  # 无折扣
```
当多头拥挤时，做多信号最多被折扣到原来的 50%。

#### FusedAlpha 输出结构

```python
@dataclass
class FusedAlpha:
    unified:          float  # 最终融合 Alpha 分数
    slow_score:       float  # 慢层原始分
    fast_boost_val:   float  # 快层 boost/drag 值
    regime_mult:      float  # 市场状态乘数
    tradability_mult: float  # 流动性乘数
    crowding_disc:    float  # 拥挤度折扣系数
    is_long_candidate:  bool  # unified > +0.30
    is_short_candidate: bool  # unified < -0.30
```

---

### 3.8 持仓生命周期状态机

**文件**：`alpha_factory/alpha_lifecycle.py`

每笔持仓在持有期间，其 unified alpha 会随市场状态变化而涨落。AlphaLifecycleTracker 通过状态机追踪这一过程，在 Alpha 反转时主动平仓。

#### 状态定义

```
BUILD      → 信号刚建立，观察期
EXPANSION  → 信号持续增强，持仓安全
DECAY      → 信号开始衰减，准备减仓
REVERSAL   → 信号已反转，立即平仓
```

#### 状态转换规则

```
BUILD → EXPANSION:
    条件：score > 0.60  AND  velocity > 0（分数高且仍在增强）

EXPANSION → DECAY:
    条件：连续 2 轮 velocity < 0（动量连续负值）

DECAY → REVERSAL:
    条件：aligned_score < -0.20（方向已反转）

任意状态 → REVERSAL:
    条件：score < -0.20（直接触发反转，跳过中间状态）
```

#### velocity 计算（EMA 速度）

```python
velocity = EMA(score_change, α=0.25)
```

每次 update 被调用（每 60 秒），velocity 由当前分数与上次分数的差值 EMA 平滑。

#### 触发平仓

在 `_rank_and_trade()` 每轮排序结束后：
```python
def _update_lifecycle_states(self, fused: dict):
    for sym in 持多仓品种:
        new_state = lifecycle_tracker.update(sym, "LONG", fused[sym].unified)
        held = now - entry_time
        if new_state == REVERSAL and held >= min_hold_seconds:
            _close_long(sym, reason="lifecycle_reversal")

    for sym in 持空仓品种:
        ...类似逻辑...
```

`min_hold_seconds`（默认 120s）防止刚进场就因短暂波动被误判 REVERSAL 而平仓。

---

### 3.9 组合构建引擎

**文件**：`portfolio/portfolio_constructor.py`

在确定候选品种后，PortfolioConstructor 解决「选哪些、各选多少」的问题。

#### 流程

**Step 1 — 候选过滤**
```python
# 过滤掉价差过宽的品种
candidates = [sym for sym in ranked if features[sym].spread_bps <= max_spread_bps]
# 过滤掉冲击暂停的品种
candidates = [sym for sym in candidates if not shock_detector.is_paused(sym)]
```

**Step 2 — 相关性去重**
```python
# 按 |unified_alpha| 降序排列候选池
# 对于每个新候选 sym_b，检查与已选组合中所有 sym_a 的相关性
corr = pearsonr(ret_1m 序列 of sym_a, ret_1m 序列 of sym_b)
if corr > 0.85:
    continue  # 相关性过高，舍弃，避免持仓重复
```

**Step 3 — Beta 中性约束**
```python
# 估算净敞口
est_long_notional  = len(selected_longs)  × base_size_usdt
est_short_notional = len(selected_shorts) × base_size_usdt
net_exposure = est_long_notional - est_short_notional

# 如果净敞口超限（默认 300 USDT），移除弱候选
while |net_exposure| > max_net_exposure:
    移除 unified_alpha 绝对值最小的一方
```

**Step 4 — Alpha 加权仓位定价**
```python
# Alpha 强度归一化
alpha_vals = [|unified_alpha| for sym in selected]
mean_alpha = mean(alpha_vals)
alpha_scale = clip(|unified_alpha| / mean_alpha, 0.7, 1.5)

# 历史 Sharpe 加权
portfolio_weight = 1.0 + 0.4 × tanh(sharpe_proxy)  # ∈ [0.5, 1.5]

# 最终仓位
target_usdt = base_size_usdt × alpha_scale × portfolio_weight
```

**输出 TargetPortfolio**
```python
@dataclass
class TargetPortfolio:
    longs:          List[TargetPosition]  # 目标多头列表
    shorts:         List[TargetPosition]  # 目标空头列表
    to_close_long:  List[str]             # 需要平仓的多头（不在目标列表中）
    to_close_short: List[str]             # 需要平仓的空头
```

---

### 3.10 交易前成本模型

**文件**：`execution/cost_model.py`

在实际下单前，对每笔交易估算完整的交易成本，只有当预期 Alpha 收益足以覆盖成本时才执行。

#### 成本构成

```
fee_cost    = notional × 0.0004 × 2       # Taker 手续费（开仓+平仓各一次）
spread_cost = notional × spread_bps / 20000  # 买卖价差损耗
impact_cost = 0.001 × (notional / depth_usdt) × notional  # 市场冲击（线性模型）

total_cost = fee_cost + spread_cost + impact_cost
```

#### 预期毛收益估算

```
expected_gross = |unified_alpha| × 0.0015 × notional
```
`0.0015`：每单位 Alpha 对应 0.15% 的预期收益，可通过回测校正。

#### 可行性判断

```python
is_viable = expected_gross >= min_edge_multiple × total_cost
# min_edge_multiple 默认 1.5
# 要求预期收益至少是成本的 1.5 倍才执行
```

不通过则记录到 `cost_reject_log` 表，并上报 ReportEngine，用于归因分析。

---

### 3.11 风控体系（三重门禁）

**文件**：`portfolio/risk_manager.py`

信号必须依次通过三道风控检验，任一失败则阻断该笔订单。

#### 第一道：仓位限制

```python
future_qty = current_qty + order_qty
if abs(future_qty) > max_position_size:
    REJECT("position_limit")
```
防止单品种仓位过度集中。

#### 第二道：频率限制

```python
# 滑动时间窗口计数过去1分钟内的下单次数
orders_in_1min = count(timestamps within last 60s)
if orders_in_1min > 600:
    REJECT("order_frequency")
```
防止程序 Bug 导致的疯狂下单。

#### 第三道：日内亏损限制

```python
if daily_pnl < max_daily_loss:  # 默认 -5000 USDT
    trading_enabled = False      # 触发熔断，全停
    REJECT("daily_loss_limit")
```

熔断后所有新订单均被拒绝，需手动调用 `risk_manager.reset()` 才能恢复交易。

每笔成交（FILL 事件）触发 PnL 更新，平多仓时：
```python
realized_pnl = (exit_price - entry_price) × qty
daily_pnl += realized_pnl
```

#### ShockDetector — 冲击检测

**文件**：`alpha_factory/shock_detector.py`

独立的异常行情监测模块：
- 检测品种级插针（单笔成交偏离移动均价 > 阈值）
- 检测全市场闪崩（多品种同时大幅波动）
- 触发后对受影响品种设置 `is_paused = True`，持续 30 秒暂停该品种交易

---

### 3.12 执行层与订单管理

#### 支持三种执行模式

| 模式 | 类 | 资金 | 用途 |
|------|-----|------|------|
| Paper Trading | `PaperEngine` | 模拟 | 逻辑验证，无资金风险 |
| Testnet | `TestnetEngine` | 假资金 | 下单代码验证，暴露精度/权限问题 |
| Live | `BinanceGateway` | 真实资金 | 实盘运行 |

#### 开多仓完整流程

```python
def _open_long(symbol, target_usdt, unified_alpha):
    # 1. 计算下单数量
    price = feature_engine.get_mid_price(symbol)
    notional = target_usdt
    qty = notional / price

    # 2. 成本模型检查
    cost_est = cost_model.estimate_from_features(symbol, notional, unified_alpha, features)
    if not cost_est.is_viable:
        report_engine.record_cost_reject(symbol, "LONG", cost_est.reject_reason)
        db.save_cost_reject(...)
        return  # 放弃

    # 3. 发送市价单
    result = execution_engine.send_order(
        symbol=symbol, side="BUY", qty=qty, order_type="MARKET"
    )

    # 4. 更新持仓记录
    long_positions[symbol] = {
        "entry_price": price, "qty": qty,
        "entry_time": time.time(), "entry_alpha": unified_alpha
    }

    # 5. 通知生命周期追踪器
    lifecycle_tracker.open_position(symbol, "LONG", unified_alpha)

    # 6. 写入数据库
    db.save_trade(symbol, "BUY", price, qty, ...)
```

#### 平多仓完整流程

```python
def _close_long(symbol, reason):
    # 1. 通知生命周期追踪器（提前注销，避免重复平仓）
    lifecycle_tracker.close_position(symbol, "LONG")

    # 2. 获取持仓信息
    pos = long_positions[symbol]

    # 3. 发送平仓单（reduce_only=True）
    result = execution_engine.send_order(
        symbol=symbol, side="SELL", qty=pos["qty"],
        order_type="MARKET", reduce_only=True
    )

    # 4. 计算并记录完整交易周期
    exit_price = feature_engine.get_mid_price(symbol)
    pnl_usdt = (exit_price - pos["entry_price"]) × pos["qty"] × leverage
    trade = TradeRecord(
        symbol=symbol, side="LONG",
        entry_price=pos["entry_price"], exit_price=exit_price,
        qty=pos["qty"], pnl_usdt=pnl_usdt,
        hold_seconds=time.time() - pos["entry_time"],
        reason=reason, factors=pos.get("factors", {})
    )
    db.save_completed_trade(trade)

    # 5. 更新报告引擎
    report_engine.record_lifecycle_exit(symbol, "LONG", lifecycle_state)

    # 6. 删除持仓记录
    del long_positions[symbol]
```

---

### 3.13 持仓管理与浮盈更新

**文件**：`portfolio/position_manager.py`

PositionManager 通过事件订阅自动维护持仓状态：

```
FILL 事件 → on_fill():
    BUY：positions[symbol].update(+qty, price)   # 加权均价更新
    SELL：positions[symbol].update(-qty, price)   # 减仓或平仓

TICK 事件 → on_tick():
    pos.unrealized_pnl = (price - entry_price) × qty
```

**Position 对象含 Alpha 生命周期追踪字段**（用于 MonitorEngine 显示和事后归因）：
```python
alpha_state:          str    # 当前生命周期状态
entry_alpha:          float  # 建仓时 unified alpha
peak_alpha:           float  # 持仓期间峰值 alpha
holding_reason:       str    # 持仓原因标签
last_lifecycle_update: float  # 最近状态更新时间戳
```

---

### 3.14 监控引擎与报告引擎

#### MonitorEngine — 实时状态打印

**文件**：`monitor/monitor_engine.py`

每 30 秒自调度打印一次，内容包含：

```
----------------------------------------------------------
[Monitor] 14:32:07  持仓快照
----------------------------------------------------------
  BTCUSDT      qty=   0.0030  entry=  45230.5000  pnl=  +12.3000  [EXPANSION ]  α=+0.847
  ETHUSDT      qty=  -0.1200  entry=   2451.8000  pnl=   -3.1200  [DECAY     ]  α=-0.423
----------------------------------------------------------
  [市场状态] regime=TRENDING          tradability=0.812  dispersion=0.00423  crowding_z=+1.23  tradeable=✓
  [候选池]   多头=['SOLUSDT', 'AVAXUSDT', ...]  空头=['XRPUSDT', ...]
  [生命周期] BUILD:2  EXPANSION:1  DECAY:1
  [过滤统计] 成本拒绝=12次  风控拒绝=3次
----------------------------------------------------------
```

#### ReportEngine — Session 级别报告

**文件**：`monitor/report_engine.py`

程序停止时生成完整的运行报告：

```
===========================================================
  Alpha Factory - Session Report
===========================================================
  运行时长  : 127.3 分钟

── 交易摘要 ────────────────────────────────────────────────
  总交易笔数: 47
  胜率      : 61.7%  (赢29 / 亏18)
  总毛盈亏  : +142.350 USDT
  总手续费  : -18.840 USDT
  净盈亏    : +123.510 USDT
  盈亏比    : 2.15x
  均持仓时长: 847s
  平均盈利  : +8.2300 USDT
  平均亏损  : -3.8100 USDT

── 退出原因分布 ──────────────────────────────────────────────
  lifecycle_reversal             28笔   59.6%
  take_profit                    11笔   23.4%
  stop_loss                       8笔   17.0%

── Alpha 生命周期退出分布 ────────────────────────────────────
  REVERSAL               28笔
  DECAY                   8笔

── 市场状态 ──────────────────────────────────────────────────
  平均可交易性: 0.743
  平均分散度  : 0.00381
  Regime 分布:
    TRENDING              87次
    MEAN_REVERTING        34次
    QUIET                 6次

── 成本过滤 ──────────────────────────────────────────────────
  成本拒绝次数: 23
    spread_too_wide            12次
    insufficient_edge           7次

── 风控统计 ──────────────────────────────────────────────────
  风控拒绝总数: 8
    position_limit              5次
    order_frequency             3次

── 因子归因（Top 5）─────────────────────────────────────────
  volume_zscore                +89.3421
  oi_change_pct                +54.1230
  funding_rate                 +31.8920
  ret_1m                       +12.3410
  ret_5m                        -8.2310
===========================================================
```

---

### 3.15 异步数据库存储

**文件**：`storage/database.py`

使用 SQLite + 后台写入线程实现非阻塞持久化：

- `on_fill_event()` / `save_trade()` 等方法只将数据放入 `Queue`（O(1) 非阻塞）
- 后台 `_writer_thread` 持续消费队列，批量写入 SQLite
- 程序退出时发送 `None` 哨兵，等待队列刷完后关闭连接

**8 张数据表**（详见第 12 节）覆盖全链路日志：成交记录、完整交易周期、Alpha 生命周期、市场状态、Alpha 融合、成本拒绝、风控拒绝、组合决策。

---

## 4. Alpha 融合公式推导

### 4.1 为什么需要融合？

慢层评分（slow_score）只反映"品种在截面中的相对强弱"，但不考虑：
- 当前市场是否适合交易（VOLATILE 时应减少交易）
- 当前流动性是否足够（价差宽时信号权重应降低）
- 快层微观结构是否支持进场（timing 不好时不应强行进场）
- 当前是否多头拥挤（拥挤时做多应有折扣）

融合公式将以上所有维度乘性组合：

```
unified = slow_score
        × regime_mult          # 市场状态适应
        × tradability_mult     # 流动性适应
        × (1 + fast_boost)     # 微结构时序增减
        × crowding_disc        # 拥挤度折扣
```

### 4.2 各项数值范围

| 项目 | 典型范围 | 极端范围 |
|------|---------|---------|
| slow_score | [-2, +2] | [-4, +4] |
| regime_mult | 0.60 ~ 1.00 | - |
| tradability_mult | 0.50 ~ 1.00 | - |
| 1 + fast_boost | 0.60 ~ 1.50 | - |
| crowding_disc | 0.50 ~ 1.00 | - |
| unified | [-2.5, +2.5] | 经 soft clip 限幅 |

### 4.3 Soft Clip 防过拟合

最终 unified 经过 soft clip：
```
unified = tanh(unified / 3.0) × 3.0
```
压制极端值，使输出始终在 [-3, +3] 内，防止单笔仓位因异常大信号而过度集中。

---

## 5. 因子工程详解

### 5.1 因子相关性与互补性

| 因子对 | 预期相关性 | 设计意图 |
|--------|----------|---------|
| volume_zscore ↔ ret_1m | 中等正相关 | 同向确认：成交量放大伴随价格动量 |
| funding_rate ↔ ret_1m | 负相关 | 反转信号：资金费率高时多头过热，均值回归 |
| oi_change_pct ↔ ret_5m | 正相关 | 趋势确认：OI 增加说明新资金入场 |
| lob_z1/z2/z3 ↔ ofi | 弱相关 | 互补：LOB 几何结构 vs 订单流动量 |

### 5.2 OFI 公式来源

Cont, Kukanov & Stoikov (2014) 的 Order Flow Imbalance 模型：
```
OFI_t = ΔB_t - ΔA_t
ΔB_t = {bid_qty_t  if bid_p_t ≥ bid_p_{t-1}} - {bid_qty_{t-1} if bid_p_t ≤ bid_p_{t-1}}
ΔA_t = {ask_qty_{t-1} if ask_p_t ≥ ask_p_{t-1}} - {ask_qty_t  if ask_p_t ≤ ask_p_{t-1}}
```

实证研究表明 OFI 对未来15秒价格变化有显著预测力（R² ≈ 0.6 for liquid assets）。

### 5.3 Microprice 理论基础

Microprice 是对 mid-price 的精炼估计，用订单簿深度加权：
```
Microprice = bid × (ask_qty / total_qty) + ask × (bid_qty / total_qty)
```
直觉：ask 侧量多时买方力量强，microprice 偏向 ask，预示价格上行。

---

## 6. LOB PCA 微观结构提取

### 6.1 为什么用 PCA？

订单簿 5 档买卖各有价格和量，共 20 维原始特征。各维度高度相关（相邻档位价格相关性接近1），直接使用会导致信息冗余和数值不稳定。PCA 提取正交的潜在因子，每个因子捕捉一个独立的结构性维度：

- **PC1**：捕捉订单簿整体形变（买卖失衡、价格冲击）
- **PC2**：捕捉价格层级变化（档位间的价格弹性）
- **PC3**：捕捉量的分布变化（薄厚档位的切换）
- **PC4**：捕捉高频噪音中的细微结构信号

### 6.2 白化的作用

原始 PCA 各主成分方差不等（PC1 解释最多），直接使用会导致 PC1 主导后续模型。白化（Whitening）通过除以各特征值的平方根，使所有主成分方差均一化为 1，后续模型可以平等对待各 PC 方向的信息。

### 6.3 分桶标准化的必要性

不同流动性状态（高频/中频/低频）下，订单簿结构截然不同：
- 高频时段：买卖盘均衡，深度充沛，价差极窄
- 低频时段：买卖盘稀疏，价差宽，各档量不稳定

如果不分桶，PCA 会在这些混合状态上拟合，提取的主成分无法稳定代表任何单一的市场结构。分桶后，每个桶的 PCA 只在同类市场状态内拟合，提取的因子更具一致性。

---

## 7. 持仓生命周期状态机详解

### 7.1 状态转换图

```
         score > 0.60 & vel > 0
  BUILD ──────────────────────────► EXPANSION
    │                                    │
    │                              vel < 0 连续2次
    │                                    ▼
    │                                  DECAY
    │                                    │
    │  score < -0.20                aligned_score < -0.20
    └──────────────────────────────────► REVERSAL ──► 触发平仓
              (任意状态直接跳转)
```

### 7.2 进场时机（BUILD 阶段）

刚进场时状态为 BUILD，系统处于观察期，不因微小波动而平仓。`min_hold_seconds`（默认120秒）确保即使立即出现 REVERSAL，也不会在 120 秒内平仓，避免因数据噪声触发假 REVERSAL。

### 7.3 与止盈止损的协同

生命周期 REVERSAL 平仓是**信号驱动的主动平仓**，而传统止盈止损是**价格驱动的被动平仓**。两者并行：
- 价格达到止损线 → 立即止损（保护资金）
- Alpha 信号反转 → lifecycle_reversal 平仓（信号驱动）
- 先触发者执行，两者相互补充

---

## 8. 组合构建与仓位定价

### 8.1 相关性去重的意义

如果两个品种都是"强势币"且收益率高度相关（比如 BTC 和 ETH 同涨同跌），同时持有两个多头仓位等于把相同的风险敞口翻倍，但 Alpha 并没有翻倍。相关性去重确保组合持有的是真正独立的 Alpha 来源。

相关性矩阵使用 `ret_1m` 序列（最近 200 个 1 分钟收益率），Pearson 相关系数 > 0.85 视为高度相关。

### 8.2 Beta 中性的实现

```
净敞口 = 多头名义总额 - 空头名义总额
目标：净敞口 ≈ 0（市场中性）
```

当净敞口超过 `max_net_exposure`（300 USDT），从较多的那一侧移除 unified_alpha 绝对值最小的候选，直到满足约束。

### 8.3 仓位定价的直觉

Alpha 越强 → alpha_scale 越高 → 单笔仓位越大（敢赌）
历史表现好 → portfolio_weight 越高 → 整体仓位规模上调（信任策略）
流动性差 → tradability_mult 低 → 进场犹豫（惜仓）

---

## 9. 风控设计

### 9.1 分层防御体系

```
Layer 0: UniverseFilter  — 从源头排除不流动品种
Layer 1: ShockDetector   — 实时检测价格冲击，暂停受影响品种
Layer 2: MarketState     — 极端市场状态全局暂停开仓
Layer 3: CostModel       — 预交易成本不可行时拦截
Layer 4: RiskManager     — 仓位/频率/日亏三重门禁
Layer 5: AlphaLifecycle  — 信号反转时主动减仓
```

每道防线各自独立，多重保护。

### 9.2 熔断机制

日内亏损熔断触发后：
1. `trading_enabled = False`
2. 所有新 SIGNAL 事件在 RiskManager 处被拦截
3. 现有持仓继续止盈止损（不强制平仓，避免在坏价格砍仓）
4. 需要人工调用 `risk_manager.reset()` 恢复（有意设计为需要人工确认）

---

## 10. 事件驱动架构

### 10.1 事件类型

| 事件 | 生产者 | 消费者 |
|------|--------|--------|
| TICK | MultiSymbolFeed | PositionManager（浮盈）、Strategy（tick 处理） |
| ORDER_BOOK | MultiSymbolFeed | Strategy（LOB 特征更新） |
| SIGNAL | Strategy | RiskManager（风控校验） |
| ORDER | RiskManager | OrderManager（下单执行） |
| FILL | BinanceGateway | PositionManager（持仓更新）、Database（写入）、RiskManager（PnL 更新） |
| ACCOUNT | BinanceGateway | AccountManager（余额同步） |
| ALPHA_UPDATE | AlphaFusionEngine | 监控/归因模块 |
| TARGET_POSITION | PortfolioConstructor | Strategy（执行目标组合） |
| RISK_REJECT | RiskManager | ReportEngine（统计） |
| KILL_SWITCH | ShockDetector | 全系统暂停 |

### 10.2 单线程单队列设计

```python
EventEngine:
    _queue = queue.Queue()
    _thread = Thread(target=_run)  # 单一消费线程

def _run():
    while True:
        event = _queue.get()
        for handler in handlers[event.type]:
            handler(event)  # 串行处理，无竞态
```

所有事件处理器在同一线程中串行执行，**完全无锁，无竞态**。复杂的并发问题转化为简单的队列顺序问题。

### 10.3 高频数据绕过事件队列

aggTrade 和 bookTicker 的 WebSocket 回调直接调用 FeatureEngine，不经过事件队列，消除额外的线程切换延迟，确保特征更新的实时性。

---

## 11. 模块目录结构

```
institutional_crypto_quant/
├── alpha_factory/               # Alpha 工厂（核心策略层）
│   ├── alpha_strategy.py        # 主策略编排器（所有引擎的协调中心）
│   ├── feature_engine.py        # 实时多因子特征计算引擎
│   ├── scoring_engine.py        # 横截面多因子评分引擎（慢层）
│   ├── lob_timing_engine.py     # LOB 微观结构时序引擎（快层）
│   ├── lob_manifold_engine.py   # LOB PCA 流形特征提取（v5）
│   ├── lob_alpha_engine.py      # LOB Alpha 信号引擎
│   ├── alpha_fusion.py          # Alpha 融合引擎（慢层×快层×市场状态）
│   ├── alpha_lifecycle.py       # 持仓 Alpha 生命周期状态机
│   ├── market_state_engine.py   # 全市场状态检测引擎
│   ├── ml_alpha_engine.py       # 机器学习 Alpha 引擎（LightGBM/在线学习）
│   ├── halflife_engine.py       # 因子半衰期估计引擎
│   ├── ranking_engine.py        # 候选品种排名引擎
│   ├── shock_detector.py        # 价格冲击检测与熔断
│   └── universe_filter.py       # 交易池动态过滤
│
├── portfolio/                   # 组合管理层
│   ├── portfolio_constructor.py # 组合构建（去重+Beta中性+仓位定价）
│   ├── position_manager.py      # 持仓状态管理（含生命周期字段）
│   ├── risk_manager.py          # 三重门禁风控系统
│   └── account_manager.py       # 账户余额管理
│
├── execution/                   # 执行层
│   ├── cost_model.py            # 预交易成本估算模型
│   └── order_manager.py         # 订单生命周期管理
│
├── exchange/                    # 交易所接入层
│   └── binance_gateway.py       # Binance 实盘下单接口
│
├── data_layer/                  # 数据层
│   ├── multi_symbol_ws.py       # 多币 WebSocket 管理
│   ├── rest_fetcher.py          # REST 数据轮询（资金费率+OI）
│   ├── websocket_client.py      # 底层 WebSocket 客户端
│   ├── order_book_live.py       # 实时订单簿管理
│   ├── data_loader.py           # 历史数据加载
│   └── logger.py                # 日志系统（日切文件）
│
├── core/                        # 核心基础设施
│   ├── event_engine.py          # 事件驱动引擎（单线程单队列）
│   ├── event.py                 # Event 数据结构
│   └── constants.py             # 事件类型枚举 + Binance 端点常量
│
├── monitor/                     # 监控与报告层
│   ├── monitor_engine.py        # 实时状态监控（30s 自调度）
│   └── report_engine.py         # Session 级别完整运行报告
│
├── storage/                     # 持久化层
│   └── database.py              # SQLite 异步写入数据库
│
├── strategy/                    # 策略基础层（已有早期策略实现）
│   ├── strategy_base.py         # 策略基类
│   └── strategy_engine.py       # 早期策略引擎
│
├── config/                      # 配置层
│   └── settings.py              # 全局配置（API Key、交易参数）
│
├── live_trading/                # 运行入口
│   ├── run_alpha_factory.py     # Paper Trading 入口
│   ├── run_alpha_factory_testnet.py  # Testnet 测试入口
│   └── main.py                  # 早期简单策略入口
│
├── dashboard/                   # 可视化仪表盘
│   └── alpha_dashboard.py       # 实时 Alpha 可视化
│
├── logs/                        # 日志目录（自动创建）
└── alpha_factory_trades.db      # SQLite 数据库文件
```

---

## 12. 数据库 Schema

系统在 `alpha_factory_trades.db` 中维护 8 张表，覆盖全链路可观测性：

### trades — 原始成交记录

| 字段 | 类型 | 含义 |
|------|------|------|
| id | INTEGER | 自增主键 |
| order_id | TEXT | 交易所订单 ID |
| symbol | TEXT | 交易对 |
| side | TEXT | BUY / SELL |
| price | REAL | 成交均价 |
| qty | REAL | 成交数量 |
| timestamp | INTEGER | 毫秒时间戳 |
| status | TEXT | FILLED / PARTIALLY_FILLED |

### completed_trades — 完整交易周期

| 字段 | 类型 | 含义 |
|------|------|------|
| symbol / side | TEXT | 品种和方向 |
| entry_price / exit_price | REAL | 开平价格 |
| qty / leverage | REAL/INT | 数量和杠杆 |
| pnl_usdt | REAL | 毛利润（USD） |
| ret_pct | REAL | 收益率（无杠杆） |
| ret_lev_pct | REAL | 收益率（含杠杆） |
| hold_seconds | REAL | 持仓时长（秒） |
| reason | TEXT | 退出原因 |
| entry_time / exit_time | INTEGER | 开平时间戳 |

### lifecycle_log — Alpha 生命周期事件

记录每次状态转换（BUILD→EXPANSION→DECAY→REVERSAL）的时间、品种、分数和速度。

### market_state_log — 市场状态快照

每次排序时的 regime、tradability、dispersion、crowding_z 全量记录。

### alpha_fusion_log — Alpha 融合快照

每轮排序的 unified、slow_score、fast_boost、各乘数的完整分解。

### cost_reject_log — 成本过滤记录

被成本模型拦截的每笔信号：品种、名义金额、总成本、预期收益、拦截原因。

### risk_reject_log — 风控拒绝记录

被风控拦截的每笔信号：品种、方向、拦截层级、原因。

### portfolio_decision_log — 组合决策记录

每次组合构建的多空数量、净敞口、待平仓数量快照。

---

## 13. 配置与部署

### 关键参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| rank_interval | 60s | 横截面评分间隔 |
| max_long_positions | 3 | 最大多头持仓数 |
| max_short_positions | 3 | 最大空头持仓数 |
| long_score_threshold | 0.50 | 做多最低 unified alpha |
| short_score_threshold | -0.50 | 做空最高 unified alpha |
| ema_alpha | 0.40 | 分数 EMA 平滑系数 |
| confirm_rounds | 2 | 连续上榜 N 次才进场 |
| leverage | 10 | 期货杠杆倍数 |
| stop_loss_pct | 0.05 | 保证金亏损 5% 止损 |
| take_profit_pct | 0.10 | 保证金盈利 10% 止盈 |
| trade_size_usdt | 100.0 | 基础仓位大小（USDT） |
| min_volume_zscore | 0.80 | 成交量 Z-score 最低门槛 |
| warmup_count | 4 | 热身期（N 轮不交易） |
| max_spread_bps | 20.0 | 最大可接受价差（基点） |
| min_hold_seconds | 120 | 最短持仓时间（防假 REVERSAL） |
| min_edge_multiple | 1.5 | 成本模型：最低净优势倍数 |

### 环境变量

```bash
# 实盘 API（真实资金）
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_api_secret

# Testnet API（假资金）
export TESTNET_API_KEY=your_testnet_key
export TESTNET_API_SECRET=your_testnet_secret
```

### 依赖安装

```bash
pip install -r requirements.txt
# 核心依赖：
# python-binance>=1.0.19
# websockets>=11.0
# numpy>=1.23
# scipy>=1.9
# pandas>=1.5
# requests>=2.28
```

---

## 14. 运行方式

### Paper Trading（推荐先验证）

```bash
cd institutional_crypto_quant
python live_trading/run_alpha_factory.py
```

- 不连接真实交易所
- 完整运行所有 Alpha 引擎、市场状态、生命周期逻辑
- 信号和仓位只在内存中记录，实时打印到控制台
- 程序退出时生成完整 Session 报告

### Testnet 测试（下单代码验证）

```bash
export TESTNET_API_KEY=your_testnet_key
export TESTNET_API_SECRET=your_testnet_secret
python live_trading/run_alpha_factory_testnet.py
```

- 行情数据使用实盘 WebSocket（数据真实）
- 下单接口连接 Binance Testnet（假资金，能暴露精度/权限问题）
- 成交记录写入本地 SQLite
- 每分钟同步查询 Testnet 真实持仓状态

### 实盘（谨慎）

```bash
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_api_secret
python live_trading/run_alpha_factory.py --live
```

**建议顺序**：Paper Trading（逻辑验证）→ Testnet（执行验证）→ 极小仓位实盘（滑点/成本验证）→ 正式实盘

### 程序优雅退出

`Ctrl+C` 触发优雅停止流程：
1. 停止所有 WebSocket 数据流
2. 停止 REST 轮询
3. 通知策略 `on_stop()`（可在此处平仓未了结持仓）
4. 生成 Session Report（ReportEngine.generate()）
5. 刷新数据库写入队列（等待后台写入线程完成）
6. 关闭 SQLite 连接
7. 打印最终汇总统计

---

*系统持续演进，以上为当前版本（终极版）的完整说明文档。*

# Institutional Crypto Quant

生产级加密货币量化交易系统，基于 Binance Futures 实现**全市场横截面 Alpha 因子策略**与**高频微观结构策略**。

---

## 目录

1. [策略定位](#1-策略定位)
2. [系统架构总览](#2-系统架构总览)
3. [模块结构](#3-模块结构)
4. [核心策略详解](#4-核心策略详解)
   - 4.1 [Alpha Factory 策略（全市场截面）](#41-alpha-factory-策略全市场截面)
   - 4.2 [HFT LOB 策略（高频微观结构）](#42-hft-lob-策略高频微观结构)
5. [完整策略流程](#5-完整策略流程)
6. [因子工程详解](#6-因子工程详解)
7. [风控体系](#7-风控体系)
8. [持仓与账户管理](#8-持仓与账户管理)
9. [Binance 交易所集成](#9-binance-交易所集成)
10. [事件驱动架构](#10-事件驱动架构)
11. [监控与日志](#11-监控与日志)
12. [配置与部署](#12-配置与部署)
13. [依赖安装](#13-依赖安装)

---

## 1. 策略定位

### 核心理念

**横截面 Alpha（Cross-Sectional Alpha）**：不预测绝对价格方向，只做「相对强弱」排序。在所有候选币中，找出「相对最强且尚未涨」的品种做多，找出「相对最弱且尚未跌」的品种做空，构成市场中性组合。

**微观结构 Alpha（Microstructure Alpha）**：利用实时订单簿（LOB）的微观结构信息，捕捉短期价格失衡和流动性冲击，实现高频信号提取。

**自适应权重（IC 进化）**：基于近期因子 IC（Information Coefficient，信息系数）动态调整各因子权重，使策略持续适应市场环境变化。

### 系统规格

| 参数 | Alpha Factory | HFT LOB |
|------|--------------|---------|
| 覆盖品种 | 270+ USDT 永续合约 | 4 个主要币种 |
| 信号频率 | 每 60 秒排序一轮 | 逐 Tick |
| 平均持仓 | 10~30 分钟 | 秒级~分钟级 |
| 杠杆 | 20x | 5x |
| 方向 | 多空双向，市场中性 | 多空双向 |
| 模式 | 纸交易 / 实盘 | 实盘 |

---

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Binance Futures                               │
│    aggTrade WebSocket · bookTicker WebSocket · @depth · User Data Stream │
│    REST API：资金费率 / 未平仓量 / 全市场品种列表 / 下单                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │           Data Layer            │
              │  MultiSymbolFeed  (WS × 5 并发) │  aggTrade + bookTicker
              │  MsOrderBookEngine (L2 订单簿)  │  @depth 增量维护
              │  RestFetcher      (60s 轮询)    │  资金费率 + OI
              └────────────────┬────────────────┘
                               │ 直接回调（零队列延迟）
              ┌────────────────▼────────────────┐
              │           Alpha Factory         │
              │                                 │
              │  FeatureEngine                  │  实时微观结构因子计算
              │  ├─ volume_zscore               │  成交量异常 Z-Score
              │  ├─ ofi / ofi_ema              │  订单流失衡（EMA 平滑）
              │  ├─ lob_z1/z2/z3              │  PCA 订单簿主成分
              │  ├─ funding_rate               │  资金费率
              │  ├─ oi_change_pct              │  未平仓量变化
              │  └─ ret_1m / ret_5m            │  短期动量
              │                                 │
              │  ScoringEngine                  │  横截面打分 + PCA 去市场因子
              │  ├─ Winsorize（±3σ 截断）       │
              │  ├─ Z-score 标准化              │
              │  ├─ 加权求和（IC 动态权重）       │
              │  └─ PCA 去共同因子              │
              │                                 │
              │  RankingEngine                  │  EMA 平滑 + 多轮确认
              │  ├─ EMA 平滑分数                │  防止单 Tick 噪声
              │  ├─ Top-N 做多候选               │
              │  ├─ Bottom-N 做空候选            │
              │  └─ confirm_rounds=2 确认门      │  连续 N 轮才开仓
              │                                 │
              │  AlphaFactoryStrategy           │  多层开平仓决策引擎
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │         Risk Manager            │
              │  Gate 1: 单币持仓上限            │
              │  Gate 2: 下单频率滑动窗口         │
              │  Gate 3: 每日亏损熔断             │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │        Execution Layer          │
              │  PaperEngine    (纸交易 stub)   │
              │  TestnetEngine  (测试网)         │
              │  BinanceGateway (实盘)           │
              │  OrderManager   (订单路由)        │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │      Portfolio + Storage        │
              │  PositionManager（持仓 P&L）     │
              │  AccountManager （余额同步）      │
              │  SQLite Database（成交持久化）    │
              │  MonitorEngine  （30s 快照打印）  │
              └─────────────────────────────────┘
```

---

## 3. 模块结构

```
institutional_crypto_quant/
│
├── alpha_factory/              核心 Alpha 引擎
│   ├── feature_engine.py         实时因子计算（aggTrade + bookTicker）
│   ├── scoring_engine.py         横截面打分（动态权重 + IC 进化 + PCA 去市场因子）
│   ├── ranking_engine.py         EMA 排序 + 多轮确认机制
│   ├── alpha_strategy.py         策略主体（信号 → 多层平仓 → 风控 → 导出）
│   ├── halflife_engine.py        均值回归半衰期计算
│   ├── lob_manifold_engine.py    PCA 订单簿流形分析
│   ├── shock_detector.py         市场冲击检测
│   └── universe_filter.py        交易池筛选（成交量 > 5M USDT/24h）
│
├── core/                       系统基础层
│   ├── constants.py              事件类型枚举 + Binance 端点常量
│   ├── event.py                  Event 数据类
│   └── event_engine.py           线程安全事件总线（单队列单消费者）
│
├── config/
│   └── settings.py               全局配置（通过环境变量覆盖，含 Testnet Key）
│
├── data_layer/                 数据接入层
│   ├── multi_symbol_ws.py        全市场 WebSocket 多路复用（5 连接并发）
│   ├── rest_fetcher.py           资金费率 + OI REST 轮询（60s 间隔）
│   ├── websocket_client.py       单品种 @trade WebSocket 基础类
│   ├── order_book_live.py        全量 + 增量 L2 订单簿维护（@depth）
│   └── logger.py                 统一日志（文件 + 控制台）
│
├── exchange/
│   └── binance_gateway.py        Binance 下单 / 账户 / User Data Stream
│
├── execution/
│   └── order_manager.py          订单路由（SIGNAL → ORDER → Gateway）
│
├── portfolio/
│   ├── position_manager.py       持仓跟踪（FILL 事件驱动，实时 P&L）
│   ├── account_manager.py        账户余额同步（ACCOUNT_UPDATE 事件）
│   └── risk_manager.py           三层风控门（持仓 / 频率 / 亏损熔断）
│
├── strategy/
│   ├── strategy_engine.py        多策略生命周期管理
│   ├── strategy_base.py          策略抽象基类
│   └── hft_lob_strategy.py       高频 LOB 微观结构策略
│
├── monitor/
│   └── monitor_engine.py         持仓快照打印（30s 间隔）
│
├── storage/
│   └── database.py               SQLite 成交记录
│
├── dashboard/
│   └── alpha_dashboard.py        Streamlit 实时可视化面板
│
├── live_trading/               启动入口
│   ├── main.py                   HFT LOB 策略启动脚本
│   ├── run_alpha_factory.py      Alpha Factory 全市场启动脚本
│   └── run_alpha_factory_testnet.py  测试网模式
│
└── requirements.txt
```

---

## 4. 核心策略详解

### 4.1 Alpha Factory 策略（全市场截面）

**文件**：`alpha_factory/alpha_strategy.py` · `live_trading/run_alpha_factory.py`

**策略目标**：在 270+ 个币种中，通过因子打分构建多空组合，捕捉币种间的相对动量差异与订单流失衡信号。

#### 策略流程

```
① 宇宙筛选（Universe Filter）
   └─ 过滤 24h 成交量 < 5M USDT 的低流动性品种
   └─ 保留 USDT 永续合约，排除分叉/下架品种

② 数据订阅
   ├─ MultiSymbolFeed：全市场 aggTrade + bookTicker WebSocket
   └─ RestFetcher：60s 轮询资金费率 + 未平仓量（OI）

③ 实时因子计算（FeatureEngine，逐 Tick 更新）
   ├─ volume_zscore：当前成交量 Z-Score（有符号，主动买=正）
   ├─ ofi_ema：订单流失衡 EMA（买方压力 - 卖方压力）
   ├─ lob_z1/z2/z3：PCA 订单簿主成分（捕捉 LOB 形态变化）
   ├─ funding_rate：当前资金费率（正=多头付空头）
   ├─ oi_change_pct：未平仓量变化率（仓位扩张/萎缩）
   └─ ret_1m / ret_5m：1 分钟 / 5 分钟价格收益率

④ 横截面打分（ScoringEngine，每轮排序时执行）
   ├─ Step 1：Winsorize 极值截断（±3σ，消除单品种噪声）
   ├─ Step 2：Z-Score 标准化（消除量纲差异）
   ├─ Step 3：因子方向调整（FACTOR_SIGNS，部分因子取反）
   ├─ Step 4：加权求和（权重由 IC 自适应更新）
   └─ Step 5：PCA 去共同因子（β × PC1 市场因子，提取纯特异性 Alpha）

⑤ 因子权重（当前版本，基于 IC 反馈动态调整）
   ├─ ofi：          0.25  （IC ≈ +0.12，反向有效）
   ├─ lob_z1：       0.25  （PCA LOB 第一主成分）
   ├─ lob_z2：       0.25  （IC ≈ +0.17，最强单因子）
   ├─ lob_z3：       0.25  （LOB 第三模态）
   ├─ volume_zscore：0.00  （IC ≈ -0.38，当前已禁用）
   ├─ oi_change_pct：0.00  （IC ≈ -0.70，OI 爆增反为崩溃信号，已禁用）
   └─ ret_1m：       0.00  （IC ≈ -0.07，噪声，已禁用）

⑥ 排序与确认（RankingEngine）
   ├─ EMA 平滑原始分数（防止 Tick 级噪声造成频繁换仓）
   ├─ 取 Top-N 做多候选 / Bottom-N 做空候选
   └─ confirm_rounds=2：连续 2 轮出现才触发开仓信号

⑦ 开平仓决策（AlphaFactoryStrategy）
   开仓条件：
   ├─ 排名持续上升且超过阈值
   ├─ 已通过确认轮次
   └─ 风控三层门全部通过

   平仓条件（多层）：
   ├─ 止损：亏损达 max_drawdown（默认 5%）
   ├─ 止盈：盈利达目标收益
   ├─ 信号反转：排名跌出做多区间 / 升出做空区间
   └─ 半衰期到期：持仓超过预测均值回归半衰期

⑧ 执行
   ├─ 纸交易模式（默认）：记录信号，不实际下单
   └─ 实盘模式（--live）：通过 BinanceGateway 提交市价单
```

---

### 4.2 HFT LOB 策略（高频微观结构）

**文件**：`strategy/hft_lob_strategy.py` · `live_trading/main.py`

**策略目标**：在 4 个核心币种上，利用 L2 订单簿微观结构信号，捕捉毫秒级价格失衡，实现高频方向性交易。

#### 策略流程

```
① 数据订阅（4 个币种，各自独立 WebSocket）
   ├─ @trade 流（BinanceWebSocketClient）：逐笔成交
   └─ @depth@100ms 流（MsOrderBookEngine）：Top-10 档 L2 实时快照

② 特征计算（FeatureEngine，每笔 Tick 更新）
   ├─ volume_zscore：当前笔成交量相对 60 笔历史的 Z-Score
   ├─ ofi：订单流失衡（Cont et al. 2014 模型）
   │       delta = bid_change × bid_qty - ask_change × ask_qty
   │       ofi_ema = 0.3 × delta + 0.7 × prev_ema
   ├─ spread_bps：买卖价差（基点），流动性代理
   ├─ depth_imbalance：(bid深度 - ask深度) / 总深度，压力指标
   ├─ ret_1m / ret_5m：短期动量
   └─ funding_rate / oi_change：衍生品情绪

③ 复合打分（ScoringEngine）
   └─ 同 Alpha Factory 流程（Winsorize → Z-Score → 加权 → PCA 去市场因子）

④ 信号生成
   ├─ 分数超过上阈值（+σ）→ 触发做多信号
   ├─ 分数低于下阈值（-σ）→ 触发做空信号
   └─ 连续 confirm_rounds=2 轮确认

⑤ 风控三层门（见第 7 节）

⑥ 执行（市价单，优先成交速度）
   └─ OrderManager → BinanceGateway → /fapi/v1/order（MARKET）

⑦ 持仓管理
   ├─ 止损：-5% drawdown 触发平仓
   └─ 止盈：+10% 触发平仓
```

---

## 5. 完整策略流程

以下以 HFT LOB 策略为例，展示一笔完整的交易从数据到成交的全链路：

```
Step 1 ── 市场数据到达
  Binance @trade 推送：ETHUSDT 成交 50 BTC @ 3500（Taker 主动买入）
  └─ BinanceWebSocketClient.on_message()
  └─ 构造 TICK 事件：{symbol:"ETHUSDT", price:3500.0, qty:50, is_buyer_maker:False}
  └─ EventEngine.put(TICK)

Step 2 ── 事件消费（串行，无竞态）
  EventEngine 后台消费线程取出 TICK 事件，依次分发给所有注册 Handler：

  a) PositionManager.on_tick()
     └─ unrealized_pnl = (3500 - entry_price) × qty → 更新浮动盈亏显示

  b) HFTLOBStrategy.on_tick()
     └─ FeatureEngine 更新 volume_zscore、ofi_ema
     └─ ScoringEngine 计算复合分数
     └─ 分数 > 阈值（且通过确认轮次）→ 发射 SIGNAL 事件
     └─ EventEngine.put(SIGNAL:{symbol:"ETHUSDT", side:"BUY", qty:0.01})

Step 3 ── 风控验证
  RiskManager.on_signal_event() 接收 SIGNAL，依次检查三层门：
  ├─ Gate 1 持仓上限：当前持仓 0.50 + 0.01 = 0.51 < 1000 ✓
  ├─ Gate 2 下单频率：过去 60s 下单 2 次 < 600 ✓
  └─ Gate 3 日亏损熔断：今日累计亏损 -1000 > -5000 USDT ✓
  全部通过 → EventEngine.put(ORDER:{symbol:"ETHUSDT", side:"BUY", qty:0.01})

Step 4 ── 订单提交
  OrderManager.on_order_event() 接收 ORDER：
  └─ 调用 BinanceGateway.place_market_order("ETHUSDT", "BUY", 0.01)
  └─ REST API → POST /fapi/v1/order（type=MARKET, side=BUY, quantity=0.01）
  └─ Binance 返回：{orderId: 123456, status: "NEW"}

Step 5 ── 成交回报
  Binance User Data Stream 推送：ORDER_TRADE_UPDATE
  └─ BinanceGateway._on_message()
  └─ 构造 FILL 事件：{symbol:"ETHUSDT", side:"BUY", price:3501.5, qty:0.01, orderId:123456}
  └─ EventEngine.put(FILL)

Step 6 ── 成交处理（串行分发）
  a) PositionManager.on_fill()
     └─ qty: 0.50 → 0.51
     └─ entry_price = (0.50 × 3499 + 0.01 × 3501.5) / 0.51 = 3499.05（加权均价）

  b) RiskManager.on_fill_event()
     └─ 本次为加仓（同向），不触发 P&L 结算，daily_pnl 不变

  c) Database.on_fill_event()
     └─ INSERT INTO trades(symbol, side, qty, price, timestamp) VALUES(...)

Step 7 ── 持仓监控（每 30 秒）
  MonitorEngine 打印：
  [10:15:30] ETHUSDT: Long 0.51 @ 3499.05, Floating PnL = +1.25 USDT
```

---

## 6. 因子工程详解

### 6.1 SymbolState 滚动缓冲区

每个币种维护独立的滚动状态（`alpha_factory/feature_engine.py`）：

| 缓冲区 | 长度 | 用途 |
|--------|------|------|
| `trade_volumes` | deque(60) | 最近 60 笔 USDT 成交量 |
| `price_series` | deque(2000) | (时间戳, 价格) 时序列 |
| `ofi_ema` | 单值 | 订单流失衡 EMA 累计值 |
| `funding_rate` | 单值 | REST 轮询最新资金费率 |
| `oi_change_pct` | 单值 | OI 变化率 |

### 6.2 各因子计算公式

**Volume Z-Score（成交量异常度）**
```
z = (current_volume - mean(volumes)) / std(volumes)
signed_z = z × (+1 if taker_buy else -1)
```
含义：主动买入大单对应正分，主动卖出大单对应负分。

**OFI EMA（订单流失衡指数）**
```
# 每次 bookTicker 更新时：
delta_bid = (new_bid_qty - old_bid_qty) if bid_price unchanged else new_bid_qty
delta_ask = (new_ask_qty - old_ask_qty) if ask_price unchanged else -new_ask_qty
delta_ofi = delta_bid - delta_ask
ofi_ema = 0.3 × delta_ofi + 0.7 × prev_ofi_ema
```
含义：正值表示买方补充深度（看涨压力），负值表示卖方扩张（看跌压力）。

**LOB PCA 主成分（lob_z1/z2/z3）**
```
LOB 特征向量 = [bid1_qty, bid2_qty, ..., bid5_qty, ask1_qty, ..., ask5_qty]
# 对历史 LOB 快照窗口做 PCA
lob_z1 = PC1 投影分数（解释最大方差，通常对应市场整体深度）
lob_z2 = PC2 投影分数（解释次要方差，通常对应买卖深度倾斜）
lob_z3 = PC3 投影分数（高阶微观结构形态）
```

**Price Returns（价格动量）**
```
ret_1m = (price_now - price_1min_ago) / price_1min_ago
ret_5m = (price_now - price_5min_ago) / price_5min_ago
# 使用二分查找在 price_series 中定位历史价格
```

### 6.3 打分流水线（ScoringEngine）

```
原始因子矩阵 [N 个币种 × M 个因子]
        ↓
1. Winsorize：将每列超出 ±3σ 的值裁剪至 ±3σ（防止单币种极端值污染截面）
        ↓
2. Z-Score 标准化：(x - mean) / std（使各因子均值=0、标准差=1，可加性增强）
        ↓
3. 因子方向对齐（FACTOR_SIGNS）：
   部分因子需取反（如 volume_zscore 若 IC 为负则乘以 -1）
        ↓
4. 加权求和：score = Σ(weight_i × z_factor_i)
   权重由 IC 自适应更新（IC > 0 → 正权重；IC < 0 → 禁用）
        ↓
5. PCA 去市场因子：
   对 score 向量做 PCA，提取第一主成分（市场公共因子 β）
   residual_score = score - β × PC1（去除系统性涨跌，保留纯特异性 Alpha）
        ↓
6. 最终归一化：(residual - mean) / std
        ↓
复合分数（均值≈0，标准差≈1，正值=相对强势，负值=相对弱势）
```

### 6.4 IC 自适应权重更新

```python
# 每轮排序后，用下一轮收益率计算因子 IC
ic = spearman_correlation(factor_scores, future_returns)
# 近 N 轮 IC 的 EWMA
ic_ema = alpha × ic_new + (1 - alpha) × ic_ema_prev
# 更新权重：仅保留 IC 显著为正的因子
weight_i = max(0, ic_ema_i) / sum(max(0, ic_ema_j) for all j)
```

---

## 7. 风控体系

### 三层风控门（RiskManager）

所有 SIGNAL 事件必须通过全部三道门才能转换为 ORDER：

```
SIGNAL 事件输入
      │
      ▼
┌─────────────────────────────────────┐
│ Gate 1：单币持仓上限                  │
│ if (current_qty + delta) > max_pos  │
│ → 拒绝，不开新仓                     │
└──────────────────┬──────────────────┘
                   │ 通过
                   ▼
┌─────────────────────────────────────┐
│ Gate 2：下单频率滑动窗口（60s）        │
│ 维护时间戳队列，清除 >60s 的记录      │
│ if count(last_60s) >= max_per_min   │
│ → 拒绝，防止 API 频率超限             │
└──────────────────┬──────────────────┘
                   │ 通过
                   ▼
┌─────────────────────────────────────┐
│ Gate 3：每日亏损熔断                  │
│ if daily_realized_pnl <= -5000 USDT │
│ → 停止全部交易，等待人工重置           │
└──────────────────┬──────────────────┘
                   │ 全部通过
                   ▼
            ORDER 事件输出
```

### 每日 P&L 追踪

RiskManager 仅在「平仓或减仓」时结算实现盈亏：

```python
# 做多平仓
realized_pnl = (fill_price - entry_price) × filled_qty

# 做空平仓
realized_pnl = (entry_price - fill_price) × filled_qty

daily_pnl += realized_pnl
```

### 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_position` | 1000 | 单币最大持仓量 |
| `max_orders_per_min` | 600 | 每分钟最大下单次数 |
| `max_daily_loss` | -5000 USDT | 每日亏损熔断阈值 |

---

## 8. 持仓与账户管理

### PositionManager（持仓追踪）

```
维护每个币种的实时持仓状态：
  qty:          净持仓数量（正=多头，负=空头，0=空仓）
  entry_price:  加权平均成本价（weighted average cost）
  unrealized:   (当前价 - 成本价) × qty（浮动盈亏）

加仓逻辑（同向增仓）：
  new_entry = (old_qty × old_price + delta_qty × fill_price) / new_qty

减仓逻辑（反向平仓）：
  entry_price 不变，仅减少 qty
  当 qty = 0 时，清空 entry_price

浮动盈亏：
  每次 TICK 事件到达时更新 unrealized_pnl = (tick_price - entry_price) × qty
```

### AccountManager（余额同步）

订阅 Binance User Data Stream 中的 `ACCOUNT_UPDATE` 事件，维护本地余额缓存：
- O(1) 查询（无 REST 轮询开销）
- 支持多资产（USDT、BTC 等）
- 为风控模块提供流动性检查依据

---

## 9. Binance 交易所集成

### BinanceGateway 双流架构

```
┌───────────────────────────────────┐
│        BinanceGateway             │
│                                   │
│  ┌─────────────────┐              │
│  │   REST Client   │              │  python-binance SDK
│  │  ├─ 市价单下单  │              │  POST /fapi/v1/order
│  │  ├─ 撤单       │              │  DELETE /fapi/v1/order
│  │  ├─ 查询订单   │              │  GET /fapi/v1/order
│  │  └─ 精度元数据 │              │  GET /fapi/v1/exchangeInfo
│  └─────────────────┘              │
│                                   │
│  ┌─────────────────────────────┐  │
│  │  User Data Stream（私有 WS）  │  │  需先获取 listenKey
│  │  ├─ 获取 listenKey（REST）  │  │  GET /fapi/v1/listenKey
│  │  ├─ Keepalive（每 30min）   │  │  PUT /fapi/v1/listenKey
│  │  ├─ 自动重连                │  │  重新获取新 listenKey
│  │  ├─ ORDER_TRADE_UPDATE →   │  │  → FILL 事件
│  │  └─ ACCOUNT_UPDATE →       │  │  → ACCOUNT 事件
│  └─────────────────────────────┘  │
└───────────────────────────────────┘
```

### 价格/数量精度处理

防止浮点误差导致 API 拒绝：

```python
# 从 exchangeInfo 缓存 stepSize（数量步长）和 tickSize（价格步长）
quantity = Decimal(qty).quantize(step_size, rounding=ROUND_DOWN)
price    = Decimal(price).quantize(tick_size, rounding=ROUND_DOWN)
```

---

## 10. 事件驱动架构

### EventEngine（事件总线）

系统核心解耦机制：**单队列 + 单消费者线程**，保证事件串行处理无竞态。

```python
# 核心结构
queue: Queue()             # 线程安全 FIFO，所有生产者写入
consumer: Thread()         # 唯一消费者，50ms 超时 blocking get
handlers: Dict[EventType, List[Callable]]  # 每种事件类型的订阅者列表
```

### 事件类型

| EventType | 生产者 | 消费者 |
|-----------|--------|--------|
| `TICK` | BinanceWebSocketClient | PositionManager, Strategy |
| `ORDER_BOOK` | MsOrderBookEngine | Strategy |
| `SIGNAL` | Strategy | RiskManager |
| `ORDER` | RiskManager | OrderManager |
| `FILL` | BinanceGateway | PositionManager, RiskManager, Database |
| `ACCOUNT` | BinanceGateway | AccountManager |

### 启动序列（HFT LOB 策略）

```
1. 验证配置（API Key、杠杆参数）
2. 创建 EventEngine（启动消费者线程）
3. 创建 BinanceGateway（建立 User Data Stream）
4. 创建 PositionManager、RiskManager、OrderManager（注册事件 Handler）
5. 创建 StrategyEngine，实例化 4 个 HFTLOBStrategy（每个币种一个）
6. 启动 MonitorEngine（每 30s 打印持仓快照）
7. 启动 WebSocket 客户端（@trade × 4 + @depth × 4）
8. signal.pause() 阻塞主线程，等待 Ctrl+C

优雅关闭（Ctrl+C → SIGINT）：
  MonitorEngine.stop() → Strategy.on_stop() → WebSocket.stop()
  → EventEngine.stop() → BinanceGateway.close() → Database.close()
```

---

## 11. 监控与日志

### MonitorEngine

每 30 秒打印实时持仓快照：

```
[10:15:30] Position Monitor
  ETHUSDT:  Qty=+1.51 @ 3499.05, Floating PnL = +25.30 USDT
  BTCUSDT:  Qty=-0.02 @ 67500.00, Floating PnL = +85.00 USDT
  ADAUSDT:  No position
  DOTUSDT:  No position
```

### Logger

结构化日志同时输出到控制台与文件（`logs/` 目录）：

```
[2026-03-28 10:15:23] [INFO]  [EventEngine]   Started, consumer thread running
[2026-03-28 10:15:25] [DEBUG] [RiskManager]   Gate1 pass: ETHUSDT qty=0.51 < 1000
[2026-03-28 10:15:26] [INFO]  [ScoringEngine] Top3: ETHUSDT:+0.85 DOTUSDT:+0.62 ...
[2026-03-28 10:15:27] [INFO]  [OrderManager]  BUY ETHUSDT 0.01 @ MARKET → orderId=123456
```

### SQLite 数据库（storage/database.py）

```sql
CREATE TABLE trades (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol    TEXT    NOT NULL,
  side      TEXT    NOT NULL,   -- 'BUY' or 'SELL'
  qty       REAL    NOT NULL,
  price     REAL    NOT NULL,
  timestamp TEXT    NOT NULL
);
```

可通过 `config.SAVE_TRADES_TO_DB = False` 禁用持久化。

### Streamlit 实时面板（dashboard/alpha_dashboard.py）

```bash
streamlit run dashboard/alpha_dashboard.py
```

展示内容：实时因子分布、各币种得分排名、历史成交记录、当前持仓。

---

## 12. 配置与部署

### 环境变量（推荐方式）

```bash
# 实盘
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_api_secret

# 测试网（内置默认值，可直接使用）
export TESTNET_API_KEY=your_testnet_key
export TESTNET_API_SECRET=your_testnet_secret
```

### 关键配置参数（config/settings.py）

```python
# 交易品种
SYMBOLS = ["ETHUSDT", "BTCUSDT", "ADAUSDT", "DOTUSDT"]

# 仓位控制
LEVERAGE            = 5       # 账户杠杆倍数
TRADE_SIZE          = 0.01    # 每笔基础开仓量（合约张数）
MAX_POSITION_SIZE   = 0.1     # 单币最大仓位（占账户比例）

# 风控
MAX_DRAWDOWN        = 0.05    # 日最大回撤 5%（触发熔断）
MAX_DAILY_LOSS      = -5000   # 日最大亏损（USDT）

# 系统
DATABASE_PATH       = "quant_trades.db"
LOG_LEVEL           = "INFO"
WEBSOCKET_TIMEOUT   = 30
SAVE_TRADES_TO_DB   = True
```

### 启动命令

```bash
# 1. Alpha Factory 纸交易（安全，无实际下单）
python live_trading/run_alpha_factory.py

# 2. Alpha Factory 实盘（需设置环境变量）
BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy \
  python live_trading/run_alpha_factory.py --live

# 3. Alpha Factory 测试网
python live_trading/run_alpha_factory_testnet.py

# 4. HFT LOB 实盘策略
python live_trading/main.py

# 5. Streamlit 监控面板
streamlit run dashboard/alpha_dashboard.py
```

---

## 13. 依赖安装

```bash
pip install -r requirements.txt
```

```
# requirements.txt
websocket-client>=1.6.0    # Binance WebSocket 连接
requests>=2.31.0           # REST API 调用
numpy>=1.24.0              # 截面因子计算（PCA、Z-Score）
python-binance>=1.0.19     # Binance 官方 SDK

# 可选（Dashboard 面板）
pandas>=2.0.0
streamlit>=1.28.0
altair>=5.0.0
```

---

## 免责声明

本项目仅供学习与研究目的。加密货币交易存在极高风险，使用本代码进行实盘交易所造成的任何损失与本项目作者无关。在投入真实资金前，请充分了解相关风险，并在测试网充分验证策略。

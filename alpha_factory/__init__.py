"""
Alpha Factory - 全市场异动捕捉系统

架构：
    UniverseFilter  → 筛选可交易币种（50~100个 USDT 永续合约）
    FeatureEngine   → 实时多因子特征计算（OFI、量异动、资金费率、OI）
    ScoringEngine   → 横截面多因子打分（Z-score 归一化）
    RankingEngine   → 全市场排序，输出 Top N 信号
    AlphaStrategy   → 执行策略（进场 / 退场 / 止损）
"""

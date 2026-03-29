"""
Alpha Factory Dashboard v2
cd institutional_crypto_quant && streamlit run dashboard/alpha_dashboard.py
"""

import json
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

# ── 路径 ───────────────────────────────────────────────────────────────────────
_BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTOR_LOG     = os.path.join(_BASE, "factor_log.csv")
IC_LOG         = os.path.join(_BASE, "ic_log.csv")
POSITIONS_FILE = os.path.join(_BASE, "positions_current.json")

FACTOR_COLS          = ["time", "symbol", "side", "factor", "value", "ret"]
IC_COLS              = ["time", "factor", "ic", "weight"]
WARMUP_SYMBOLS_TEXT  = "15"    # 与 lob_manifold_engine.WARMUP_SYMBOLS 保持一致

# ── 颜色 ───────────────────────────────────────────────────────────────────────
G   = "#00C896"   # green
R   = "#FF4560"   # red
B   = "#3D8EF8"   # blue
Y   = "#FEB019"   # yellow
P   = "#775DD0"   # purple
DIM = "#546E7A"   # dim gray
BG2 = "#161B2E"   # card bg

PALETTE = [B, G, Y, P, R, "#00D4FF", "#FF6B35", "#A8FF3E"]

# ── 页面配置 ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha Factory",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 0; }}

/* 卡片 */
[data-testid="metric-container"] {{
    background: {BG2};
    border: 1px solid #252A3D;
    border-radius: 10px;
    padding: 14px 20px 12px;
}}
[data-testid="stMetricValue"] {{ font-size: 1.55rem !important; font-weight: 700; }}
[data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}

/* 分组标题 */
.sec {{ font-size:0.68rem; font-weight:700; letter-spacing:.14em;
        text-transform:uppercase; color:#6B7A99; margin:1.4rem 0 .6rem; }}

/* 状态徽章 */
.badge-g {{ background:#0D3326; color:{G}; border:1px solid #1A5E42;
            border-radius:5px; padding:2px 10px; font-size:.75rem; font-weight:600; }}
.badge-r {{ background:#2D1218; color:{R}; border:1px solid #5A1E28;
            border-radius:5px; padding:2px 10px; font-size:.75rem; font-weight:600; }}
.badge-b {{ background:#0E1D3A; color:{B}; border:1px solid #1E3D7A;
            border-radius:5px; padding:2px 10px; font-size:.75rem; font-weight:600; }}

/* 数据表格 */
.stDataFrame {{ border-radius:8px; overflow:hidden; }}
div[data-testid="stDataFrame"] > div {{ border-radius:8px; }}

/* tab 样式 */
button[data-baseweb="tab"] {{ font-size:.85rem; font-weight:600; }}

hr {{ border-color:#252A3D; margin:.8rem 0; }}
</style>
""", unsafe_allow_html=True)


# ── Altair 主题 ────────────────────────────────────────────────────────────────
def _reg_theme():
    def _t():
        return {"config": {
            "background": "transparent",
            "axis":   {"gridColor": "#252A3D", "domainColor": "#252A3D",
                       "labelColor": "#8899AA", "titleColor": "#AAB8CC",
                       "tickColor": "#252A3D", "labelFontSize": 11},
            "legend": {"labelColor": "#CCDDEE", "titleColor": "#8899AA",
                       "labelFontSize": 11},
            "view":   {"stroke": "transparent"},
            "title":  {"color": "#CCDDEE", "fontSize": 12},
        }}
    alt.themes.register("q", _t)
    alt.themes.enable("q")

_reg_theme()


# ── 数据加载 ───────────────────────────────────────────────────────────────────
def _csv(path, cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path, names=cols, header=0)
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        return df.dropna(subset=["time"])
    except Exception:
        return pd.DataFrame(columns=cols)


def _positions():
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _empty(msg="暂无数据，等待策略运行…"):
    st.markdown(f'<div style="color:#6B7A99;font-size:.88rem;padding:20px 0 8px;">⏳ {msg}</div>',
                unsafe_allow_html=True)


# ── 加载 ───────────────────────────────────────────────────────────────────────
ic_df     = _csv(IC_LOG, IC_COLS)
factor_df = _csv(FACTOR_LOG, FACTOR_COLS)
positions = _positions()

if not factor_df.empty:
    factor_df["ret"]   = pd.to_numeric(factor_df["ret"],   errors="coerce")
    factor_df["value"] = pd.to_numeric(factor_df["value"], errors="coerce")
    factor_df = factor_df.dropna(subset=["ret", "value"])

if not ic_df.empty:
    ic_df["ic"]     = pd.to_numeric(ic_df["ic"],     errors="coerce")
    ic_df["weight"] = pd.to_numeric(ic_df["weight"], errors="coerce")

# 基础统计
trades = pd.DataFrame()
if not factor_df.empty:
    trades = factor_df.groupby(["time", "symbol", "side"])["ret"].first().reset_index()

n_trades = len(trades)
win_rate = (trades["ret"] > 0).mean() * 100 if n_trades else 0
avg_ret  = trades["ret"].mean() * 100      if n_trades else 0
total_ret= trades["ret"].sum()             if n_trades else 0
n_long   = len(positions.get("long",  [])) if positions else 0
n_short  = len(positions.get("short", [])) if positions else 0
net_exp  = positions.get("net_exposure", 0) if positions else 0


# ══════════════════════════════════════════════════════════════════════════════
# 顶栏
# ══════════════════════════════════════════════════════════════════════════════
top_l, top_r = st.columns([6, 1])
with top_l:
    st.markdown("## ⚡ Alpha Factory")
    st.markdown(
        f'<span style="color:#6B7A99;font-size:.82rem;">'
        f'Binance Futures · 横截面 LOB 流形策略 · '
        f'{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}'
        f'</span>', unsafe_allow_html=True
    )
with top_r:
    if st.button("🗑️ 清空数据", use_container_width=True):
        for f in [FACTOR_LOG, IC_LOG, POSITIONS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.toast("数据已清空")
        time.sleep(0.6)
        st.rerun()

st.markdown("---")

# ── 顶部指标行 ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("总交易笔数", f"{n_trades}")
c2.metric("胜率", f"{win_rate:.1f}%",
          delta="盈利" if win_rate >= 50 else "亏损",
          delta_color="normal" if win_rate >= 50 else "inverse")
c3.metric("平均单笔", f"{avg_ret:+.2f}%",
          delta="每笔均值")
c4.metric("累计毛收益", f"{total_ret*100:+.2f}%")
c5.metric("持仓 多/空", f"{n_long} / {n_short}")
c6.metric("净敞口", f"{net_exp:+.0f} U",
          delta="中性" if abs(net_exp) < 30 else ("偏多" if net_exp > 0 else "偏空"),
          delta_color="off")

lob_ready = bool(positions.get("lob_ready")) if positions else False
c7.metric("LOB 流形", "就绪 ✓" if lob_ready else "热身中…",
          delta=f"PC2/3/4 激活" if lob_ready else "等待样本积累",
          delta_color="normal" if lob_ready else "off")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📈  收益总览", "🔬  因子分析", "📊  持仓监控", "🧠  LOB 流形"])


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1：收益总览
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    if trades.empty:
        _empty("等待有平仓记录后显示…")
    else:
        # 权益曲线
        st.markdown('<div class="sec">权益曲线（累计毛收益率）</div>', unsafe_allow_html=True)
        eq = trades.sort_values("time").copy()
        eq["cum_ret"] = eq["ret"].cumsum() * 100
        eq["color"]   = eq["ret"].apply(lambda x: "正收益" if x >= 0 else "负收益")

        equity_line = (
            alt.Chart(eq)
            .mark_line(color=B, strokeWidth=2, interpolate="step-after")
            .encode(
                x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
                y=alt.Y("cum_ret:Q", title="累计收益率 (%)"),
                tooltip=[alt.Tooltip("time:T", format="%H:%M:%S"),
                         alt.Tooltip("cum_ret:Q", format="+.2f", title="累计 %")],
            )
        )
        zero_rule = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color=DIM, strokeDash=[4, 4], opacity=0.5)
            .encode(y="y:Q")
        )
        area = (
            alt.Chart(eq)
            .mark_area(opacity=0.12, color=B, interpolate="step-after")
            .encode(
                x="time:T",
                y=alt.Y("cum_ret:Q"),
            )
        )
        st.altair_chart((area + equity_line + zero_rule).properties(height=240),
                        use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="sec">收益率分布</div>', unsafe_allow_html=True)
            hist = (
                alt.Chart(trades)
                .mark_bar(binSpacing=1, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("ret:Q", bin=alt.Bin(maxbins=35), title="单笔收益率"),
                    y=alt.Y("count():Q", title="笔数"),
                    color=alt.condition(
                        alt.datum.ret > 0, alt.value(G), alt.value(R)
                    ),
                    tooltip=[alt.Tooltip("ret:Q", bin=True, format=".3f"), "count():Q"],
                )
                .properties(height=220)
            )
            st.altair_chart(hist, use_container_width=True)

        with col_b:
            st.markdown('<div class="sec">平仓原因分布</div>', unsafe_allow_html=True)
            if "reason" in factor_df.columns:
                reasons = (
                    factor_df.groupby(["time", "symbol", "side"])
                    .agg(reason=("ret", "first"))
                    .reset_index()
                )
            # factor_df 里没有 reason 列，从 positions 里拿或提示
            _empty("平仓原因数据需在 factor_log 中记录 reason 字段")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2：因子分析
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if ic_df.empty and factor_df.empty:
        _empty("等待策略运行并产生因子记录（约 10 轮排序后）…")
    else:
        # ── IC 时间序列 + 权重进化 ────────────────────────────────────────────
        st.markdown('<div class="sec">因子 IC 时间序列</div>', unsafe_allow_html=True)

        if not ic_df.empty:
            all_factors = sorted(ic_df["factor"].dropna().unique())

            # 过滤掉一直是 0 的因子（LOB 热身期）
            active = ic_df.groupby("factor")["ic"].apply(lambda x: x.abs().max())
            active_factors = active[active > 0.001].index.tolist()
            lob_inactive   = [f for f in all_factors if f not in active_factors]

            sel = st.multiselect(
                "选择因子", all_factors,
                default=active_factors,
                key="ic_sel",
            )
            sub_ic = ic_df[ic_df["factor"].isin(sel)] if sel else ic_df

            col_l, col_r = st.columns(2)

            with col_l:
                st.caption("IC（Pearson 相关系数）")
                ic_data = sub_ic.dropna(subset=["ic"])
                if not ic_data.empty:
                    chart_ic = (
                        alt.Chart(ic_data)
                        .mark_line(interpolate="monotone", strokeWidth=2.5)
                        .encode(
                            x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
                            y=alt.Y("ic:Q", title="IC",
                                    scale=alt.Scale(zero=False),
                                    axis=alt.Axis(format="+.2f")),
                            color=alt.Color("factor:N",
                                            scale=alt.Scale(range=PALETTE),
                                            legend=alt.Legend(orient="bottom")),
                            tooltip=["factor:N",
                                     alt.Tooltip("time:T", format="%H:%M:%S"),
                                     alt.Tooltip("ic:Q", format="+.4f")],
                        )
                        .properties(height=240)
                    )
                    zero = (alt.Chart(pd.DataFrame({"y": [0]}))
                            .mark_rule(color=DIM, strokeDash=[4, 4], opacity=0.5)
                            .encode(y="y:Q"))
                    st.altair_chart(chart_ic + zero, use_container_width=True)

            with col_r:
                st.caption("因子权重进化")
                wt_data = sub_ic.dropna(subset=["weight"])
                if not wt_data.empty:
                    chart_wt = (
                        alt.Chart(wt_data)
                        .mark_line(interpolate="monotone", strokeWidth=2.5)
                        .encode(
                            x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
                            y=alt.Y("weight:Q", title="权重",
                                    scale=alt.Scale(zero=True)),
                            color=alt.Color("factor:N",
                                            scale=alt.Scale(range=PALETTE),
                                            legend=alt.Legend(orient="bottom")),
                            tooltip=["factor:N",
                                     alt.Tooltip("time:T", format="%H:%M:%S"),
                                     alt.Tooltip("weight:Q", format=".4f")],
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(chart_wt, use_container_width=True)

            # ── 最新因子 IC 快照表 ─────────────────────────────────────────────
            st.markdown('<div class="sec">最新因子指标</div>', unsafe_allow_html=True)
            latest = (
                ic_df.sort_values("time")
                .groupby("factor")[["ic", "weight"]]
                .last()
                .reset_index()
                .sort_values("ic", ascending=False)
            )
            latest["方向"] = latest["ic"].apply(
                lambda v: "🟢 正向" if v > 0.05 else ("🔴 反向" if v < -0.05 else "⚪ 弱")
            )
            if lob_inactive:
                latest["方向"] = latest.apply(
                    lambda r: "⏳ 热身中" if r["factor"] in lob_inactive else r["方向"],
                    axis=1
                )
            st.dataframe(
                latest.rename(columns={"factor": "因子", "ic": "IC", "weight": "权重"})
                .style
                .format({"IC": "{:+.4f}", "权重": "{:.4f}"})
                .background_gradient(subset=["IC"], cmap="RdYlGn", vmin=-0.4, vmax=0.4)
                .hide(axis="index"),
                use_container_width=True,
            )

        # ── 因子散点 + 五分位 ──────────────────────────────────────────────────
        if not factor_df.empty:
            st.markdown('<div class="sec">因子预测力（Z-score vs 收益）</div>', unsafe_allow_html=True)

            avail = sorted(factor_df["factor"].dropna().unique())
            chosen = st.selectbox("选择因子", avail, key="fsel")
            sub_f  = factor_df[factor_df["factor"] == chosen].copy()
            ic_val = sub_f["value"].corr(sub_f["ret"])

            m1, m2, m3 = st.columns([1, 1, 5])
            badge = ("badge-g" if (ic_val or 0) > 0.05
                     else ("badge-r" if (ic_val or 0) < -0.05 else "badge-b"))
            m1.metric("全周期 IC", f"{ic_val:+.4f}" if ic_val is not None else "N/A")
            m2.metric("样本量", f"{len(sub_f)} 笔")

            col1, col2 = st.columns([3, 2])

            with col1:
                st.caption("因子 Z-score vs 收益率（散点）")
                sample = sub_f.sample(min(600, len(sub_f)), random_state=42)
                scatter = (
                    alt.Chart(sample)
                    .mark_circle(size=36, opacity=0.5)
                    .encode(
                        x=alt.X("value:Q", title="因子 Z-score"),
                        y=alt.Y("ret:Q",   title="收益率"),
                        color=alt.condition(
                            alt.datum.ret > 0, alt.value(B), alt.value(R)
                        ),
                        tooltip=["symbol:N", "side:N",
                                 alt.Tooltip("value:Q", format=".3f"),
                                 alt.Tooltip("ret:Q",   format="+.4f")],
                    )
                    .properties(height=300)
                )
                zh = (alt.Chart(pd.DataFrame({"y": [0]}))
                      .mark_rule(color=DIM, strokeDash=[4, 4], opacity=0.4)
                      .encode(y="y:Q"))
                zv = (alt.Chart(pd.DataFrame({"x": [0]}))
                      .mark_rule(color=DIM, strokeDash=[4, 4], opacity=0.4)
                      .encode(x="x:Q"))
                st.altair_chart(scatter + zh + zv, use_container_width=True)

            with col2:
                st.caption("五分位平均收益（Q5 应最高）")
                try:
                    sub_f["q"] = pd.qcut(sub_f["value"], q=5,
                                          labels=["Q1","Q2","Q3","Q4","Q5"],
                                          duplicates="drop")
                    qr = sub_f.groupby("q", observed=True)["ret"].mean().reset_index()
                    bar = (
                        alt.Chart(qr)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(
                            x=alt.X("q:N", sort=None, title=None),
                            y=alt.Y("ret:Q", title="平均收益",
                                    axis=alt.Axis(format="+.3f")),
                            color=alt.condition(
                                alt.datum.ret > 0, alt.value(G), alt.value(R)
                            ),
                            tooltip=["q:N", alt.Tooltip("ret:Q", format="+.4f")],
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(bar, use_container_width=True)
                except Exception:
                    _empty("数据量不足，无法计算五分位")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3：持仓监控
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    if not positions:
        _empty("等待策略输出持仓快照（positions_current.json）…")
    else:
        ts_str = (
            pd.Timestamp(positions.get("ts", 0), unit="s").strftime("%H:%M:%S")
            if positions.get("ts") else "N/A"
        )

        col_info, col_exp = st.columns([4, 1])
        with col_info:
            st.markdown(f'<div class="sec">持仓快照  <span style="color:#6B7A99;font-size:.75rem;">更新于 {ts_str}</span></div>',
                        unsafe_allow_html=True)
        with col_exp:
            direction = "中性" if abs(net_exp) < 30 else ("偏多 ▲" if net_exp > 0 else "偏空 ▼")
            color = DIM if abs(net_exp) < 30 else (G if net_exp > 0 else R)
            st.markdown(
                f'<div style="text-align:right;padding-top:1.4rem;">'
                f'<span style="color:{color};font-size:.88rem;font-weight:600;">'
                f'净敞口 {net_exp:+.1f} U  {direction}</span></div>',
                unsafe_allow_html=True,
            )

        col_l, col_s = st.columns(2)
        _fmt = {"entry_price": "{:.5g}", "score": "{:+.3f}",
                "ret_lev_pct": "{:+.2f}%", "held_s": "{:.0f}s"}

        def _pos_table(records, side_color):
            if not records:
                st.markdown(f'<div style="color:#6B7A99;font-size:.85rem;padding:8px 0;">无持仓</div>',
                            unsafe_allow_html=True)
                return
            df = pd.DataFrame(records)
            show_cols = [c for c in ["symbol", "entry_price", "score", "ret_lev_pct", "held_s"] if c in df.columns]
            df = df[show_cols]
            styled = df.style.format(
                {k: v for k, v in _fmt.items() if k in df.columns}
            )
            if "ret_lev_pct" in df.columns:
                styled = styled.applymap(
                    lambda v: (f"color:{G};font-weight:600" if isinstance(v, (int, float)) and v > 0
                               else (f"color:{R};font-weight:600" if isinstance(v, (int, float)) and v < 0 else "")),
                    subset=["ret_lev_pct"],
                )
            st.dataframe(styled.hide(axis="index"), use_container_width=True)

        with col_l:
            longs = positions.get("long", [])
            st.markdown(f'<span style="color:{G};font-weight:700;font-size:.9rem;">▲ 多头 ({len(longs)})</span>',
                        unsafe_allow_html=True)
            _pos_table(longs, G)

        with col_s:
            shorts = positions.get("short", [])
            st.markdown(f'<span style="color:{R};font-weight:700;font-size:.9rem;">▼ 空头 ({len(shorts)})</span>',
                        unsafe_allow_html=True)
            _pos_table(shorts, R)

        # 因子权重条形图
        weights = positions.get("weights", {})
        if weights:
            st.markdown('<div class="sec">当前运行时因子权重</div>', unsafe_allow_html=True)
            wt_df = (
                pd.DataFrame([(k, v) for k, v in weights.items() if v > 0],
                             columns=["factor", "weight"])
                .sort_values("weight", ascending=True)
            )
            is_lob = wt_df["factor"].str.startswith("lob")
            wt_df["color"] = wt_df["factor"].apply(
                lambda f: P if f.startswith("lob") else B
            )
            bar_wt = (
                alt.Chart(wt_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X("weight:Q", title="权重"),
                    y=alt.Y("factor:N", sort=None, title=None),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=["factor:N", alt.Tooltip("weight:Q", format=".4f")],
                )
                .properties(height=max(120, len(wt_df) * 28))
            )
            st.altair_chart(bar_wt, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4：LOB 流形
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec">LOB 流形引擎状态</div>', unsafe_allow_html=True)

    lob_status = positions.get("lob_status", {}) if positions else {}

    if lob_status:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("PCA 状态",    "就绪 ✓" if lob_status.get("eigen_fitted") else "热身中")
        s2.metric("已跟踪品种",   lob_status.get("symbols_tracked", "-"))
        s3.metric("已产生信号",   lob_status.get("symbols_latent",  "-"))
        warmup_pct = lob_status.get("warmup_progress", 0) * 100
        s4.metric("热身进度",     f"{warmup_pct:.0f}%")

        if warmup_pct < 100:
            st.progress(warmup_pct / 100, text=f"等待 {WARMUP_SYMBOLS_TEXT} 个品种积累聚合 delta…")
    else:
        st.info("LOB 引擎状态未写入 positions_current.json，热身进度未知。\n\n"
                "策略重启后，约 15~20 秒可看到 PC2/PC3/PC4 因子激活。")

    st.markdown('<div class="sec">LOB 因子 IC</div>', unsafe_allow_html=True)

    if not ic_df.empty:
        lob_ic = ic_df[ic_df["factor"].str.startswith("lob")].copy()
        if lob_ic.empty or lob_ic["ic"].abs().max() < 0.001:
            _empty("LOB 因子尚在热身期，所有值为 0。等待约 15 秒后 PC2/PC3/PC4 激活…")
        else:
            chart_lob = (
                alt.Chart(lob_ic.dropna(subset=["ic"]))
                .mark_line(interpolate="monotone", strokeWidth=2.5)
                .encode(
                    x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
                    y=alt.Y("ic:Q", title="IC", axis=alt.Axis(format="+.2f"),
                            scale=alt.Scale(zero=False)),
                    color=alt.Color("factor:N",
                                    scale=alt.Scale(
                                        domain=["lob_z1", "lob_z2", "lob_z3"],
                                        range=[B, G, Y]
                                    ),
                                    legend=alt.Legend(orient="bottom")),
                    tooltip=["factor:N",
                             alt.Tooltip("time:T", format="%H:%M:%S"),
                             alt.Tooltip("ic:Q", format="+.4f")],
                )
                .properties(height=240, title="LOB 流形因子 IC（PC2/PC3/PC4，已去除市场方向 PC1）")
            )
            zero = (alt.Chart(pd.DataFrame({"y": [0]}))
                    .mark_rule(color=DIM, strokeDash=[4, 4], opacity=0.5)
                    .encode(y="y:Q"))
            st.altair_chart(chart_lob + zero, use_container_width=True)
    else:
        _empty("等待 IC 数据…")

    # 原理说明
    st.markdown('<div class="sec">设计原理</div>', unsafe_allow_html=True)
    st.markdown("""
| 步骤 | 操作 | 原因 |
|------|------|------|
| **标准化 LOB** | 价格 → 偏中价率，量 → /总深度 | 消除不同币绝对规模差异 |
| **ΔLOB / dt** | 除以真实时间间隔 | 变化速度，跨品种可比 |
| **滚动聚合** | 最近 5 个 velocity 取均值 | 降噪，提升 IC 稳定性 |
| **EWMA 协方差** | 时间衰减半衰期 60s | 实时追踪市场结构，无需重拟合 |
| **丢弃 PC1** | 只用 PC2/PC3/PC4 | PC1 ≈ 市场整体方向（beta），去掉后为纯 idiosyncratic alpha |
| **EMA 平滑输出** | α = 0.25 | 低通滤波，稳定信号 |
""")


# ── 自动刷新 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 设置")
    auto_refresh = st.toggle("自动刷新", value=False)
    if auto_refresh:
        secs = st.slider("刷新间隔（秒）", 10, 120, 30)
        time.sleep(secs)
        st.rerun()

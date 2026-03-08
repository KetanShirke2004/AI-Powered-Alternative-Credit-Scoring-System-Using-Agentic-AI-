"""
Applicant Dashboard — Portfolio View with Multiple Applicants
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.data_utils import generate_synthetic_applicant, compute_credit_score


@st.cache_data
def build_portfolio(n: int = 20) -> list:
    portfolio = []
    for i in range(n):
        app = generate_synthetic_applicant(seed=i * 17)
        result = compute_credit_score(app)
        portfolio.append({
            "id": f"APP-{app['SK_ID_CURR']}",
            "name": f"Applicant {i+1:02d}",
            "income": app["AMT_INCOME_TOTAL"],
            "loan": app["AMT_CREDIT"],
            "employment": app["NAME_INCOME_TYPE"],
            "years_employed": app["YEARS_EMPLOYED"],
            "education": app["NAME_EDUCATION_TYPE"],
            "gender": app["CODE_GENDER"],
            "age": app["AGE_YEARS"],
            "score": result["score"],
            "tier": result["risk_tier"],
            "tier_color": result["risk_color"],
            "approval_prob": result["approval_probability"],
            "default_prob": result["default_probability"],
            "max_loan": result["max_loan_amount"],
            "rate": result["recommended_rate"],
            "ext_avg": round((app["EXT_SOURCE_1"] + app["EXT_SOURCE_2"] + app["EXT_SOURCE_3"]) / 3, 3),
            "on_time": app["ON_TIME_PAYMENTS_PCT"],
            "bureau_overdue": app["BUREAU_OVERDUE_DEBT"],
            "has_car": app["FLAG_OWN_CAR"],
            "has_realty": app["FLAG_OWN_REALTY"],
        })
    return portfolio


def render():
    st.markdown("""
    <div class="section-header">
        <div class="section-eyebrow">◈ LOAN PORTFOLIO</div>
        <div class="section-title">Applicant Dashboard</div>
        <div class="section-subtitle">
            Real-time portfolio view of loan applicants with AI credit scores and risk tiers
        </div>
    </div>
    """, unsafe_allow_html=True)

    portfolio = build_portfolio(20)
    df = pd.DataFrame(portfolio)

    PLOTLY_THEME = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B9EC7", family="DM Sans"),
        title_font=dict(color="#F0F4FF", family="Syne"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    )

    # Portfolio summary
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#00D4FF;">{len(df)}</div>
            <div class="metric-label">Total Applicants</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        approved = len(df[df["approval_prob"] >= 0.6])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#00FF88;">{approved}</div>
            <div class="metric-label">Likely Approved</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#FFB800;">{df['score'].mean():.0f}</div>
            <div class="metric-label">Avg Credit Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        total_exposure = df["loan"].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#7B61FF;">₹{total_exposure/1e6:.1f}M</div>
            <div class="metric-label">Total Exposure</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        avg_default = df["default_prob"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#FF4560;">{avg_default:.1%}</div>
            <div class="metric-label">Portfolio Default Risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tier_filter = st.multiselect(
            "Risk Tier", df["tier"].unique().tolist(),
            default=df["tier"].unique().tolist()
        )
    with col_f2:
        min_score, max_score = st.slider("Score Range", 300, 850, (300, 850))
    with col_f3:
        emp_filter = st.multiselect(
            "Employment", df["employment"].unique().tolist(),
            default=df["employment"].unique().tolist()
        )

    filtered = df[
        (df["tier"].isin(tier_filter)) &
        (df["score"] >= min_score) & (df["score"] <= max_score) &
        (df["employment"].isin(emp_filter))
    ]

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        # Score scatter with loan amount
        fig = px.scatter(
            filtered, x="score", y="loan",
            size="income", color="tier",
            color_discrete_map={
                "EXCELLENT": "#00FF88", "GOOD": "#7BFF00",
                "FAIR": "#FFB800", "POOR": "#FF7043", "VERY POOR": "#FF4560"
            },
            hover_data=["name", "approval_prob", "rate"],
            title="Credit Score vs Loan Amount",
            labels={"score": "Credit Score", "loan": "Loan Amount (₹)"}
        )
        fig.update_layout(**PLOTLY_THEME, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_ch2:
        # Risk distribution gauge
        tier_counts = filtered["tier"].value_counts()
        tier_order = ["EXCELLENT", "GOOD", "FAIR", "POOR", "VERY POOR"]
        colors = {"EXCELLENT": "#00FF88", "GOOD": "#7BFF00", "FAIR": "#FFB800",
                  "POOR": "#FF7043", "VERY POOR": "#FF4560"}

        fig2 = go.Figure()
        x_pos = 0
        for tier in tier_order:
            if tier in tier_counts.index:
                count = tier_counts[tier]
                fig2.add_trace(go.Bar(
                    x=[count], y=["Portfolio"],
                    orientation="h",
                    name=tier,
                    marker_color=colors[tier],
                    text=f"{tier}<br>{count}",
                    textposition="inside",
                    textfont=dict(size=10, color="white"),
                    hovertemplate=f"{tier}: {count} applicants<extra></extra>"
                ))

        fig2.update_layout(
            barmode="stack",
            title="Portfolio Risk Composition",
            **PLOTLY_THEME,
            height=320,
            xaxis_title="Number of Applicants",
            legend=dict(font=dict(size=9), orientation="h", y=-0.25)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Applicant cards grid
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#00D4FF;
                letter-spacing:0.15em; margin:1.5rem 0 1rem;">APPLICANT PROFILES</div>
    """, unsafe_allow_html=True)

    # Sort options
    sort_by = st.selectbox(
        "Sort by", ["score", "loan", "income", "approval_prob", "default_prob"],
        index=0
    )
    sort_asc = st.checkbox("Ascending order", value=False)
    filtered_sorted = filtered.sort_values(sort_by, ascending=sort_asc)

    # Display as cards (3 per row)
    rows = [filtered_sorted.iloc[i:i+3] for i in range(0, min(len(filtered_sorted), 12), 3)]

    for row_df in rows:
        cols = st.columns(3)
        for col, (_, app) in zip(cols, row_df.iterrows()):
            with col:
                tier_colors_map = {
                    "EXCELLENT": ("#00FF88", "rgba(0,255,136,0.12)", "rgba(0,255,136,0.3)"),
                    "GOOD":      ("#7BFF00", "rgba(123,255,0,0.12)",  "rgba(123,255,0,0.3)"),
                    "FAIR":      ("#FFB800", "rgba(255,184,0,0.12)",  "rgba(255,184,0,0.3)"),
                    "POOR":      ("#FF7043", "rgba(255,112,67,0.12)", "rgba(255,112,67,0.3)"),
                    "VERY POOR": ("#FF4560", "rgba(255,69,96,0.12)",  "rgba(255,69,96,0.3)"),
                }
                tc, tc_bg, tc_border = tier_colors_map.get(app["tier"], ("#00D4FF", "rgba(0,212,255,0.12)", "rgba(0,212,255,0.3)"))

                score_pct      = int((app["score"] - 300) / 5.5)
                approval_pct   = f"{app['approval_prob']:.0%}"
                income_k       = f"{app['income']/1000:.0f}K"
                loan_k         = f"{app['loan']/1000:.0f}K"
                employment_s   = str(app["employment"])[:15]
                rate_s         = str(app["rate"])[:12]
                badge_style = "display:inline-flex;align-items:center;padding:3px 10px;border-radius:100px;font-size:0.7rem;font-weight:600;background:rgba(0,212,255,0.12);color:#00D4FF;border:1px solid rgba(0,212,255,0.3);"
                badges = []
                if app["has_car"] == "Y":
                    badges.append(f'<span style="{badge_style}">&#128663; Car</span>')
                if app["has_realty"] == "Y":
                    badges.append(f'<span style="{badge_style}">&#127968; Property</span>')
                ot_color  = "#00FF88" if app["on_time"] >= 90 else "#FFB800"
                ot_bg     = "rgba(0,255,136,0.12)" if app["on_time"] >= 90 else "rgba(255,184,0,0.12)"
                ot_border = "rgba(0,255,136,0.3)"  if app["on_time"] >= 90 else "rgba(255,184,0,0.3)"
                on_time_val = int(app["on_time"])
                badges.append(f'<span style="display:inline-flex;align-items:center;padding:3px 10px;border-radius:100px;font-size:0.7rem;font-weight:600;background:{ot_bg};color:{ot_color};border:1px solid {ot_border};">&#10003; {on_time_val}% on-time</span>')
                badges_html = " ".join(badges)

                html = f"""
<div style="background:#111827;border:1px solid rgba(0,212,255,0.12);border-radius:16px;padding:1.25rem;margin-bottom:0.75rem;font-family:'DM Sans',sans-serif;">

  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.75rem;">
    <div>
      <div style="font-weight:700;font-size:0.95rem;color:#F0F4FF;">{app["name"]}</div>
      <div style="font-size:0.68rem;color:#4A5568;font-family:monospace;">{app["id"]}</div>
    </div>
    <span style="padding:3px 10px;border-radius:100px;font-size:0.68rem;font-weight:700;letter-spacing:0.05em;background:{tc_bg};color:{tc};border:1px solid {tc_border};">{app["tier"]}</span>
  </div>

  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
    <div style="font-family:monospace;font-size:2rem;font-weight:700;color:{tc};">{app["score"]}</div>
    <div style="flex:1;">
      <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:100px;overflow:hidden;">
        <div style="width:{score_pct}%;height:100%;background:{tc};border-radius:100px;"></div>
      </div>
      <div style="font-size:0.65rem;color:#8B9EC7;margin-top:4px;">Approval: {approval_pct}</div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;font-size:0.75rem;margin-bottom:0.75rem;">
    <div style="color:#8B9EC7;">Income</div>
    <div style="color:#F0F4FF;font-family:monospace;text-align:right;">&#8377;{income_k}</div>
    <div style="color:#8B9EC7;">Loan Ask</div>
    <div style="color:#F0F4FF;font-family:monospace;text-align:right;">&#8377;{loan_k}</div>
    <div style="color:#8B9EC7;">Employment</div>
    <div style="color:#F0F4FF;font-size:0.7rem;text-align:right;">{employment_s}</div>
    <div style="color:#8B9EC7;">Rate</div>
    <div style="color:#FFB800;font-size:0.7rem;text-align:right;">{rate_s}</div>
  </div>

  <div style="display:flex;gap:4px;flex-wrap:wrap;">
    {badges_html}
  </div>

</div>
"""
                st.markdown(html, unsafe_allow_html=True)

                if st.button("View Full Analysis", key=f"view_{app['id']}", use_container_width=True):
                    full_app = generate_synthetic_applicant(seed=list(filtered_sorted.index).index(app.name) * 17)
                    st.session_state["current_applicant"] = full_app
                    st.session_state["score_result"] = compute_credit_score(full_app)
                    st.session_state.current_page = "agentic_analysis"
                    st.rerun()
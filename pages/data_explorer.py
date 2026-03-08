"""
Data Explorer Page — Home Credit Dataset Analysis
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data_utils import generate_dataset_sample


@st.cache_data
def get_dataset():
    return generate_dataset_sample(n=500)


def render():
    st.markdown("""
    <div class="section-header">
        <div class="section-eyebrow">◈ HOME CREDIT DATASET</div>
        <div class="section-title">Data Explorer</div>
        <div class="section-subtitle">
            Interactive analysis of the Home Credit Default Risk dataset structure and patterns.
            Simulated from the real Kaggle dataset (307,511 applications · 122 features).
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = get_dataset()

    # Dataset stats
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#00D4FF;">{len(df):,}</div>
            <div class="metric-label">Sample Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#7B61FF;">{df.shape[1]}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        default_rate = df["TARGET"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#FF4560;">{default_rate:.1%}</div>
            <div class="metric-label">Default Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#00FF88;">{df['credit_score'].mean():.0f}</div>
            <div class="metric-label">Avg Credit Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#FFB800;">₹{df['income'].median()/1000:.0f}K</div>
            <div class="metric-label">Median Income</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distributions", "🔗 Correlations", "🎯 Default Analysis", "📋 Raw Data"
    ])

    PLOTLY_THEME = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B9EC7", family="DM Sans"),
        title_font=dict(color="#F0F4FF", family="Syne"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    )

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            # Credit score distribution
            fig = go.Figure()
            for tier, color in [("EXCELLENT", "#00FF88"), ("GOOD", "#7BFF00"),
                                 ("FAIR", "#FFB800"), ("POOR", "#FF7043"), ("VERY POOR", "#FF4560")]:
                tier_data = df[df["risk_tier"] == tier]["credit_score"]
                if len(tier_data) > 0:
                    fig.add_trace(go.Histogram(
                        x=tier_data, name=tier, marker_color=color,
                        opacity=0.7, nbinsx=20
                    ))
            fig.update_layout(
                title="Credit Score Distribution by Risk Tier",
                barmode="overlay",
                **PLOTLY_THEME,
                legend=dict(font=dict(size=10, color="#8B9EC7")),
                height=320
            )
            st.plotly_chart(fig, use_container_width=True)

            # Income distribution
            fig2 = px.histogram(
                df, x="income", nbins=40,
                color_discrete_sequence=["#00D4FF"],
                title="Income Distribution"
            )
            fig2.update_layout(**PLOTLY_THEME, height=280)
            fig2.update_traces(opacity=0.75)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            # Risk tier breakdown (donut)
            tier_counts = df["risk_tier"].value_counts()
            fig3 = go.Figure(go.Pie(
                labels=tier_counts.index,
                values=tier_counts.values,
                hole=0.55,
                marker=dict(colors=["#00FF88", "#7BFF00", "#FFB800", "#FF7043", "#FF4560"]),
                textinfo="label+percent",
                textfont=dict(size=11, color="white"),
            ))
            fig3.update_layout(
                title="Risk Tier Distribution",
                **PLOTLY_THEME,
                height=300,
                showlegend=False,
                annotations=[dict(text=f"{len(df)}<br>apps", x=0.5, y=0.5,
                                  font=dict(size=14, color="white", family="Space Mono"), showarrow=False)]
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Employment type breakdown
            emp_counts = df["employment_type"].value_counts().head(6)
            fig4 = go.Figure(go.Bar(
                x=emp_counts.values, y=emp_counts.index,
                orientation="h",
                marker=dict(
                    color=emp_counts.values,
                    colorscale=[[0, "#0D1421"], [1, "#00D4FF"]]
                ),
            ))
            fig4.update_layout(
                title="Employment Type Breakdown",
                **PLOTLY_THEME,
                height=280,
                showlegend=False
            )
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        st.markdown("""
        <div style="font-size:0.85rem; color:#8B9EC7; margin-bottom:1rem;">
            Correlation between key features and default risk (TARGET=1 means default)
        </div>
        """, unsafe_allow_html=True)

        num_cols = ["credit_score", "income", "loan_amount", "approval_probability",
                    "ext_source_1", "ext_source_2", "ext_source_3",
                    "on_time_payments_pct", "annuity_income_ratio",
                    "years_employed", "age", "bureau_overdue_debt", "TARGET"]

        corr = df[num_cols].corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#FF4560"], [0.5, "#111827"], [1, "#00D4FF"]],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
        ))
        fig_corr.update_layout(
            title="Feature Correlation Matrix",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8B9EC7", family="DM Sans"),
            title_font=dict(color="#F0F4FF", family="Syne"),
            height=520,
            xaxis=dict(tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            # Credit score by default
            fig_box = go.Figure()
            for target, label, color in [(0, "No Default", "#00FF88"), (1, "Default", "#FF4560")]:
                data = df[df["TARGET"] == target]["credit_score"]
                # Convert hex color to rgba for fillcolor (Plotly doesn't support 8-digit hex)
                hex_to_rgba = {
                    "#00FF88": "rgba(0,255,136,0.18)",
                    "#FF4560": "rgba(255,69,96,0.18)",
                }
                fill = hex_to_rgba.get(color, "rgba(128,128,128,0.18)")
                fig_box.add_trace(go.Box(
                    y=data, name=label, marker_color=color,
                    boxmean="sd", fillcolor=fill,
                    line=dict(color=color)
                ))
            fig_box.update_layout(
                title="Credit Score: Default vs No Default",
                **PLOTLY_THEME, height=320
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Default rate by employment
            default_by_emp = df.groupby("employment_type")["TARGET"].mean().sort_values(ascending=False)
            fig_emp = go.Figure(go.Bar(
                x=default_by_emp.index,
                y=default_by_emp.values * 100,
                marker=dict(
                    color=default_by_emp.values,
                    colorscale=[[0, "#00FF88"], [1, "#FF4560"]]
                ),
                text=[f"{v:.1f}%" for v in default_by_emp.values * 100],
                textposition="outside",
                textfont=dict(size=10, color="white")
            ))
            fig_emp.update_layout(
                title="Default Rate by Employment Type (%)",
                **PLOTLY_THEME, height=300
            )
            st.plotly_chart(fig_emp, use_container_width=True)

        with col_d2:
            # Scatter: Income vs Loan Amount colored by default
            fig_scatter = px.scatter(
                df.sample(min(300, len(df))),
                x="income", y="loan_amount",
                color="TARGET",
                color_discrete_map={0: "#00D4FF", 1: "#FF4560"},
                opacity=0.6,
                title="Income vs Loan Amount (colored by Default)",
                labels={"income": "Annual Income (₹)", "loan_amount": "Loan Amount (₹)", "TARGET": "Default"},
                size_max=6
            )
            fig_scatter.update_layout(**PLOTLY_THEME, height=320)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Default rate by risk tier
            default_by_tier = df.groupby("risk_tier")["TARGET"].agg(["mean", "count"]).reset_index()
            default_by_tier.columns = ["risk_tier", "default_rate", "count"]
            tier_order = ["VERY POOR", "POOR", "FAIR", "GOOD", "EXCELLENT"]
            default_by_tier["risk_tier"] = pd.Categorical(default_by_tier["risk_tier"], categories=tier_order, ordered=True)
            default_by_tier = default_by_tier.sort_values("risk_tier")
            colors = ["#FF4560", "#FF7043", "#FFB800", "#7BFF00", "#00FF88"]

            fig_tier = go.Figure(go.Bar(
                x=default_by_tier["risk_tier"],
                y=default_by_tier["default_rate"] * 100,
                marker_color=colors[:len(default_by_tier)],
                text=[f"{v:.0f}%" for v in default_by_tier["default_rate"] * 100],
                textposition="outside", textfont=dict(size=10, color="white")
            ))
            fig_tier.update_layout(
                title="Actual Default Rate by Risk Tier (%)",
                **PLOTLY_THEME, height=300
            )
            st.plotly_chart(fig_tier, use_container_width=True)

    with tab4:
        st.markdown("""
        <div style="font-size:0.8rem; color:#8B9EC7; margin-bottom:0.75rem;">
            Showing 500-record synthetic sample with Home Credit dataset structure
        </div>
        """, unsafe_allow_html=True)

        cols_show = ["applicant_id", "age", "income", "loan_amount", "employment_type",
                     "credit_score", "risk_tier", "approval_probability",
                     "ext_source_2", "on_time_payments_pct", "TARGET"]

        filter_tier = st.multiselect(
            "Filter by Risk Tier",
            df["risk_tier"].unique().tolist(),
            default=df["risk_tier"].unique().tolist()
        )

        filtered_df = df[df["risk_tier"].isin(filter_tier)][cols_show]
        st.dataframe(
            filtered_df.style.background_gradient(subset=["credit_score"], cmap="RdYlGn"),
            use_container_width=True,
            height=400
        )

        col_dl1, col_dl2 = st.columns([3, 1])
        with col_dl2:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "⬇ Download CSV",
                csv, "credit_sample.csv", "text/csv",
                use_container_width=True
            )
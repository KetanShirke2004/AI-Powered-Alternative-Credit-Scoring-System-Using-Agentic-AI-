"""
Home Page — CreditAI System Overview
"""

import streamlit as st


def render():
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="section-eyebrow">◈ AI-Powered Financial Inclusion</div>
        <div class="hero-title">Alternative Credit<br>Scoring for Everyone</div>
        <div class="hero-subtitle">
            Empowering 1.4 billion unbanked individuals with fair, AI-driven credit assessment
            using alternative data — transaction patterns, digital footprint, utility payments,
            and behavioral signals.
        </div>
        <div style="display:flex; gap:12px; flex-wrap:wrap;">
            <span class="badge badge-success">✓ Home Credit Dataset</span>
            <span class="badge badge-info">◈ 5-Agent AI Pipeline</span>
            <span class="badge badge-info">⚡ Real-time Scoring</span>
            <span class="badge badge-warning">⚖ Fair Lending AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4, col5 = st.columns(5)
    stats = [
        ("307,511", "Loan Applications", "#00D4FF"),
        ("122", "Feature Signals", "#7B61FF"),
        ("0.784", "Model AUC-ROC", "#00FF88"),
        ("5", "AI Agents", "#FFB800"),
        ("8.18%", "Default Rate", "#FF4560"),
    ]
    for col, (val, label, color) in zip([col1, col2, col3, col4, col5], stats):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two columns: Pipeline + Problem Statement
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class="section-header">
            <div class="section-eyebrow">SYSTEM ARCHITECTURE</div>
            <div class="section-title" style="font-size:1.5rem">5-Agent AI Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

        pipeline_steps = [
            ("01", "📥 Data Collection Agent",
             "Validates applicant data, identifies gaps, and enriches alternative signals from digital footprint, mobile, and utility sources."),
            ("02", "🔍 Alt Data Specialist Agent",
             "Interprets non-traditional credit signals: EXT_SOURCE scores, behavioral patterns, document compliance, and asset ownership."),
            ("03", "🎯 Risk Assessment Agent",
             "Computes default probability using ML ensemble (LightGBM), SHAP explanations, and multi-factor risk decomposition."),
            ("04", "⚖️ Decision Agent",
             "Makes final APPROVE/CONDITIONAL/DECLINE recommendation with interest rate, loan amount, and actionable conditions."),
            ("05", "🛡️ Compliance Guard Agent",
             "Audits the decision for demographic bias, fair lending compliance, and explainability under regulatory standards."),
        ]

        for num, title, desc in pipeline_steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">{num}</div>
                <div class="step-content">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="section-header">
            <div class="section-eyebrow">PROBLEM STATEMENT</div>
            <div class="section-title" style="font-size:1.5rem">The Financial Inclusion Gap</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="credit-card" style="margin-bottom:1rem;">
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#00D4FF; letter-spacing:0.15em; margin-bottom:0.75rem;">THE CHALLENGE</div>
            <p style="color:#8B9EC7; font-size:0.88rem; line-height:1.7;">
            <strong style="color:#F0F4FF;">1.4 billion adults</strong> globally lack access to formal financial services.
            Traditional credit scoring excludes them because they lack credit history — not because they're
            untrustworthy borrowers.
            </p>
            <p style="color:#8B9EC7; font-size:0.88rem; line-height:1.7; margin-top:0.75rem;">
            Existing models are rigid, opaque, and built on FICO-era assumptions that systematically
            disadvantage informal workers, migrants, students, and rural populations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="credit-card" style="margin-bottom:1rem;">
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#00FF88; letter-spacing:0.15em; margin-bottom:0.75rem;">OUR SOLUTION</div>
            <p style="color:#8B9EC7; font-size:0.88rem; line-height:1.7;">
            CreditAI combines <strong style="color:#F0F4FF;">Home Credit's alternative dataset</strong> (307K applicants)
            with a 5-agent LLM pipeline powered by Claude. Each agent specializes in one dimension
            of creditworthiness assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Alternative data sources
        st.markdown("""
        <div class="credit-card">
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#7B61FF; letter-spacing:0.15em; margin-bottom:0.75rem;">ALTERNATIVE DATA SIGNALS</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
        """, unsafe_allow_html=True)

        signals = [
            "📱 Mobile usage patterns",
            "💳 Transaction history",
            "📧 Digital presence",
            "🏠 Property ownership",
            "⚡ Utility payments",
            "📄 Document compliance",
            "👔 Employment stability",
            "🌍 Geographic signals",
        ]
        for signal in signals:
            st.markdown(f"""
            <div style="background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.1);
                        border-radius:8px; padding:6px 10px; font-size:0.78rem; color:#8B9EC7;">
                {signal}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset info
    st.markdown("""
    <div class="section-header">
        <div class="section-eyebrow">DATASET</div>
        <div class="section-title" style="font-size:1.5rem">Home Credit Default Risk (Kaggle)</div>
        <div class="section-subtitle">307,511 loan applications · 122 features · 8 data tables · 2.7 GB</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    dataset_cards = [
        ("application_train.csv", "Main training file with target variable (DEFAULT=1/0)", "307,511 rows", "#00D4FF"),
        ("bureau.csv", "Credit Bureau history — previous loans from other institutions", "1.7M rows", "#7B61FF"),
        ("previous_application.csv", "Previous Home Credit applications and their outcomes", "1.67M rows", "#00FF88"),
        ("installments_payments.csv", "Repayment history for previous loans — behavioral signal", "13.6M rows", "#FFB800"),
    ]
    for col, (name, desc, size, color) in zip([col1, col2, col3, col4], dataset_cards):
        with col:
            st.markdown(f"""
            <div class="credit-card" style="text-align:center; height:160px;">
                <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:{color}; margin-bottom:0.5rem;">
                    {name}
                </div>
                <div style="font-size:0.78rem; color:#8B9EC7; line-height:1.5; margin-bottom:0.5rem;">
                    {desc}
                </div>
                <div style="font-family:'Space Mono',monospace; font-size:0.85rem; color:{color}; font-weight:700;">
                    {size}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CTA
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        st.markdown("""
        <div style="text-align:center; padding:2rem; background:var(--bg-card);
                    border:1px solid var(--border); border-radius:20px;">
            <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800;
                        color:#F0F4FF; margin-bottom:0.5rem;">
                Ready to assess creditworthiness?
            </div>
            <div style="color:#8B9EC7; margin-bottom:1.5rem; font-size:0.9rem;">
                Run the full 5-agent AI pipeline on any applicant profile
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧠 Start AI Credit Assessment →", type="primary", use_container_width=True):
            st.session_state.current_page = "credit_assessment"
            st.rerun()

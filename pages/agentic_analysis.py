"""
Agentic Analysis Page — Full 5-Agent AI Pipeline
"""

import streamlit as st
import time
from utils.data_utils import generate_synthetic_applicant, compute_credit_score
from agents.credit_agents import CreditScoringOrchestrator


def render():
    st.markdown("""
    <div class="section-header">
        <div class="section-eyebrow">◈ MULTI-AGENT AI PIPELINE</div>
        <div class="section-title">Agentic Credit Analysis</div>
        <div class="section-subtitle">
            5 specialized AI agents collaborate to deliver a comprehensive, explainable credit decision.
            Each agent focuses on a specific dimension of creditworthiness.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Agent overview cards
    agents_info = [
        ("📥", "DataCollector", "Data Validation", "#00D4FF", "Validates completeness, flags anomalies"),
        ("🔍", "AltDataSpecialist", "Alternative Data", "#00FF88", "Interprets non-traditional signals"),
        ("🎯", "RiskAnalyst", "Risk Assessment", "#7B61FF", "Computes default probability & factors"),
        ("⚖️", "DecisionMaker", "Credit Decision", "#FFB800", "APPROVE / CONDITIONAL / DECLINE"),
        ("🛡️", "ComplianceGuard", "Fairness Review", "#FF4560", "Bias detection & explainability"),
    ]

    cols = st.columns(5)
    for col, (icon, name, role, color, desc) in zip(cols, agents_info):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; padding:1rem 0.5rem;">
                <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:{color};
                            letter-spacing:0.1em; margin-bottom:0.25rem;">{name}</div>
                <div style="font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:700;
                            color:#F0F4FF; margin-bottom:0.4rem;">{role}</div>
                <div style="font-size:0.7rem; color:#8B9EC7; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Applicant selection
    col_sel, col_run = st.columns([3, 1])
    with col_sel:
        applicant_mode = st.radio(
            "Applicant Data Source",
            ["Use Current Assessment Profile", "Generate New Sample", "Manual Entry (Quick)"],
            horizontal=True
        )

    if applicant_mode == "Use Current Assessment Profile":
        applicant = st.session_state.get("current_applicant", generate_synthetic_applicant(seed=1))
        score_result = st.session_state.get("score_result", compute_credit_score(applicant))
    elif applicant_mode == "Generate New Sample":
        import random
        seed = st.slider("Sample Seed", 0, 9999, random.randint(0, 9999))
        applicant = generate_synthetic_applicant(seed=seed)
        score_result = compute_credit_score(applicant)
    else:
        # Quick manual
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            q_income = st.number_input("Income (₹)", 10000, 2000000, 60000, 5000)
            q_loan = st.number_input("Loan Amount (₹)", 10000, 2000000, 150000, 10000)
        with col_q2:
            q_ext2 = st.slider("Alt Credit Score (EXT2)", 0.0, 1.0, 0.6, 0.01)
            q_on_time = st.slider("On-time Payments %", 0, 100, 90)
        with col_q3:
            q_years = st.slider("Years Employed", 0, 40, 5)
            q_overdue = st.number_input("Bureau Overdue (₹)", 0, 500000, 0, 1000)

        applicant = generate_synthetic_applicant(seed=42)
        applicant.update({
            "AMT_INCOME_TOTAL": q_income,
            "AMT_CREDIT": q_loan,
            "AMT_ANNUITY": q_loan * 0.08,
            "EXT_SOURCE_2": q_ext2,
            "EXT_SOURCE_1": q_ext2 * 0.9,
            "EXT_SOURCE_3": q_ext2 * 1.05,
            "ON_TIME_PAYMENTS_PCT": q_on_time,
            "YEARS_EMPLOYED": q_years,
            "BUREAU_OVERDUE_DEBT": q_overdue,
            "ANNUITY_INCOME_RATIO": (q_loan * 0.08) / max(q_income, 1),
        })
        score_result = compute_credit_score(applicant)

    # Applicant summary card
    st.markdown(f"""
    <div class="credit-card" style="background:rgba(0,212,255,0.04); margin-bottom:1.5rem;">
        <div style="display:grid; grid-template-columns:repeat(6,1fr); gap:1rem; text-align:center;">
            <div>
                <div style="font-size:0.65rem; color:#8B9EC7; letter-spacing:0.1em; margin-bottom:4px;">CREDIT SCORE</div>
                <div style="font-family:'Space Mono',monospace; font-size:1.4rem; color:{score_result['risk_color']}; font-weight:700;">
                    {score_result['score']}
                </div>
            </div>
            <div>
                <div style="font-size:0.65rem; color:#8B9EC7; letter-spacing:0.1em; margin-bottom:4px;">RISK TIER</div>
                <div style="font-family:'Syne',sans-serif; font-size:0.9rem; color:{score_result['risk_color']}; font-weight:700;">
                    {score_result['risk_tier']}
                </div>
            </div>
            <div>
                <div style="font-size:0.65rem; color:#8B9EC7; letter-spacing:0.1em; margin-bottom:4px;">INCOME</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.9rem; color:#F0F4FF;">
                    ₹{applicant.get('AMT_INCOME_TOTAL', 0):,.0f}
                </div>
            </div>
            <div>
                <div style="font-size:0.65rem; color:#8B9EC7; letter-spacing:0.1em; margin-bottom:4px;">LOAN ASKED</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.9rem; color:#F0F4FF;">
                    ₹{applicant.get('AMT_CREDIT', 0):,.0f}
                </div>
            </div>
            <div>
                <div style="font-size:0.65rem; color:#8B9EC7; letter-spacing:0.1em; margin-bottom:4px;">DEFAULT PROB</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.9rem; color:#FF4560;">
                    {score_result['default_probability']:.1%}
                </div>
            </div>
            <div>
                <div style="font-size:0.65rem; color:#8B9EC7; letter-spacing:0.1em; margin-bottom:4px;">APPROVAL PROB</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.9rem; color:#00FF88;">
                    {score_result['approval_probability']:.0%}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Run pipeline button
    run_pipeline = st.button(
        "🚀 Launch Full 5-Agent Pipeline",
        type="primary",
        use_container_width=True
    )

    if run_pipeline or "agent_results" in st.session_state:
        if run_pipeline:
            # Run with live progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            orchestrator = CreditScoringOrchestrator()
            results_container = {}

            def progress_cb(step, total, msg):
                pct = int(step / total * 100)
                progress_bar.progress(pct)
                status_text.markdown(f"""
                <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#00D4FF; padding:4px 0;">
                    ⚡ {msg}
                </div>
                """, unsafe_allow_html=True)

            with st.spinner(""):
                results = orchestrator.run_pipeline(
                    applicant, score_result,
                    progress_callback=progress_cb
                )

            st.session_state["agent_results"] = results
            progress_bar.progress(100)
            status_text.markdown("""
            <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#00FF88; padding:4px 0;">
                ✓ Pipeline complete — all 5 agents finished
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

        results = st.session_state.get("agent_results", {})

        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#00D4FF;
                    letter-spacing:0.15em; margin:1.5rem 0 1rem;">AGENT OUTPUTS</div>
        """, unsafe_allow_html=True)

        # Display agent results
        order = ["data_analysis", "alt_data_analysis", "risk_analysis", "decision", "compliance"]

        for key in order:
            if key in results:
                r = results[key]
                # Special styling for decision
                border_color = "#FFB800" if key == "decision" else "#00D4FF" if key == "compliance" else "rgba(0,212,255,0.12)"

                with st.expander(f"{r['icon']} {r['title']} — {r['agent']}", expanded=(key in ["decision", "risk_analysis"])):
                    st.markdown(f"""
                    <div style="border-left:3px solid {r['color']}; padding-left:1rem; margin-bottom:0.5rem;">
                        <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:{r['color']};
                                    letter-spacing:0.15em; margin-bottom:0.5rem;">
                            {r['agent'].upper()} · {r['timestamp']}
                        </div>
                        <div style="font-size:0.88rem; color:#C8D4F0; line-height:1.75; white-space:pre-wrap;">
                            {r['output']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Pipeline summary
        st.markdown("---")
        col_sum1, col_sum2, col_sum3 = st.columns(3)

        with col_sum1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color:#00FF88;">5/5</div>
                <div class="metric-label">Agents Completed</div>
            </div>
            """, unsafe_allow_html=True)

        with col_sum2:
            decision_text = results.get("decision", {}).get("output", "")
            if "APPROVE" in decision_text.upper() and "DECLINE" not in decision_text.upper()[:50]:
                decision_display = "APPROVED"
                d_color = "#00FF88"
            elif "CONDITIONAL" in decision_text.upper():
                decision_display = "CONDITIONAL"
                d_color = "#FFB800"
            else:
                decision_display = "DECLINED"
                d_color = "#FF4560"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{d_color}; font-size:1.2rem;">{decision_display}</div>
                <div class="metric-label">AI Recommendation</div>
            </div>
            """, unsafe_allow_html=True)

        with col_sum3:
            compliance_text = results.get("compliance", {}).get("output", "")
            flags = compliance_text.upper().count("FLAG") + compliance_text.upper().count("CONCERN")
            flag_color = "#00FF88" if flags == 0 else "#FFB800" if flags <= 2 else "#FF4560"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{flag_color};">{flags}</div>
                <div class="metric-label">Compliance Flags</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    height:300px; border:1px dashed rgba(0,212,255,0.15); border-radius:20px;
                    text-align:center; padding:2rem; margin-top:1rem;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🤖</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:#8B9EC7;">
                Launch the pipeline to activate all 5 AI agents
            </div>
            <div style="font-size:0.82rem; color:#4A5568; margin-top:0.5rem;">
                Each agent will analyze the applicant and pass findings to the next
            </div>
        </div>
        """, unsafe_allow_html=True)

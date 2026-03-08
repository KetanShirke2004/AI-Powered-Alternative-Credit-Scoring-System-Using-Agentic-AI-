"""
Credit Assessment Page — Interactive loan application scoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from utils.data_utils import generate_synthetic_applicant, compute_credit_score
from agents.credit_agents import CreditScoringOrchestrator


def render_score_gauge(score: int, color: str) -> str:
    """Render a premium animated SVG credit score gauge."""
    import math

    # ── Geometry ──────────────────────────────────────────────────────────────
    cx, cy, r_outer, r_inner = 160, 155, 120, 88
    stroke_w = r_outer - r_inner                          # 32px track width

    def arc_point(cx, cy, r, deg):
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    def arc_path(cx, cy, r, start_deg, end_deg, w):
        """Thick arc as a stroked path."""
        x1, y1 = arc_point(cx, cy, r, start_deg)
        x2, y2 = arc_point(cx, cy, r, end_deg)
        sweep  = 1 if end_deg > start_deg else 0
        large  = 1 if abs(end_deg - start_deg) > 180 else 0
        return (f'<path d="M {x1:.2f},{y1:.2f} A {r},{r} 0 {large},{sweep} {x2:.2f},{y2:.2f}" '
                f'fill="none" stroke-width="{w}" stroke-linecap="round"/>')

    # Arc spans 210° — from 195° to 345° (bottom-left → bottom-right)
    START_DEG, END_DEG = 195, 345
    RANGE_DEG = END_DEG - START_DEG                       # 150°

    # Active arc end angle based on score
    pct       = (score - 300) / 550
    active_end = START_DEG + pct * RANGE_DEG

    # Needle tip
    needle_r  = r_inner - 8
    nx, ny    = arc_point(cx, cy, needle_r, active_end)

    # Risk tier labels  (angle → label)
    ticks = [
        (START_DEG,             "300", "#FF4560"),
        (START_DEG + 0.27*RANGE_DEG, "580", "#FFB800"),
        (START_DEG + 0.55*RANGE_DEG, "660", "#7BFF00"),
        (START_DEG + 0.82*RANGE_DEG, "750", "#00FF88"),
        (END_DEG,               "850", "#00E5FF"),
    ]

    # ── SVG ───────────────────────────────────────────────────────────────────
    W, H = 320, 210
    mid_r = (r_outer + r_inner) / 2

    svg_parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                 f'xmlns="http://www.w3.org/2000/svg">']

    # ── Defs ──────────────────────────────────────────────────────────────────
    svg_parts.append("""<defs>
  <linearGradient id="trackGrad" x1="0%" y1="0%" x2="100%" y2="0%" gradientUnits="userSpaceOnUse"
      x1="40" y1="155" x2="280" y2="155">
    <stop offset="0%"   stop-color="#FF4560"/>
    <stop offset="30%"  stop-color="#FF7043"/>
    <stop offset="55%"  stop-color="#FFB800"/>
    <stop offset="75%"  stop-color="#7BFF00"/>
    <stop offset="100%" stop-color="#00E5FF"/>
  </linearGradient>
  <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
    <feGaussianBlur stdDeviation="4" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="softglow" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="8" result="blur"/>
    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="innershadow">
    <feOffset dx="0" dy="2"/>
    <feGaussianBlur stdDeviation="3" result="blur"/>
    <feComposite in="SourceGraphic" in2="blur" operator="over"/>
  </filter>
  <radialGradient id="hubGrad" cx="50%" cy="50%" r="50%">
    <stop offset="0%"   stop-color="#2A3550"/>
    <stop offset="100%" stop-color="#0D1117"/>
  </radialGradient>
  <radialGradient id="bgGlow" cx="50%" cy="65%" r="50%">
    <stop offset="0%"   stop-color="{color}" stop-opacity="0.08"/>
    <stop offset="100%" stop-color="#0D1117" stop-opacity="0"/>
  </radialGradient>
</defs>""".replace("{color}", color))

    # Background glow
    svg_parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="145" ry="110" fill="url(#bgGlow)"/>')

    # ── Track background (dim full arc) ───────────────────────────────────────
    x1s, y1s = arc_point(cx, cy, mid_r, START_DEG)
    x2s, y2s = arc_point(cx, cy, mid_r, END_DEG)
    svg_parts.append(
        f'<path d="M {x1s:.2f},{y1s:.2f} A {mid_r},{mid_r} 0 0,1 {x2s:.2f},{y2s:.2f}" '
        f'fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="{stroke_w}" stroke-linecap="round"/>'
    )

    # ── Coloured active arc ────────────────────────────────────────────────────
    x1a, y1a = arc_point(cx, cy, mid_r, START_DEG)
    x2a, y2a = arc_point(cx, cy, mid_r, active_end)
    large_a   = 1 if (active_end - START_DEG) > 180 else 0
    # Shadow/glow layer
    svg_parts.append(
        f'<path d="M {x1a:.2f},{y1a:.2f} A {mid_r},{mid_r} 0 {large_a},1 {x2a:.2f},{y2a:.2f}" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_w + 10}" '
        f'stroke-linecap="round" opacity="0.25" filter="url(#softglow)"/>'
    )
    # Main arc
    svg_parts.append(
        f'<path d="M {x1a:.2f},{y1a:.2f} A {mid_r},{mid_r} 0 {large_a},1 {x2a:.2f},{y2a:.2f}" '
        f'fill="none" stroke="url(#trackGrad)" stroke-width="{stroke_w}" '
        f'stroke-linecap="round" opacity="0.95"/>'
    )

    # ── Segment tick marks ────────────────────────────────────────────────────
    for tick_deg, label, tcol in ticks:
        tx_o, ty_o = arc_point(cx, cy, r_outer + 4, tick_deg)
        tx_i, ty_i = arc_point(cx, cy, r_inner - 4, tick_deg)
        svg_parts.append(
            f'<line x1="{tx_i:.1f}" y1="{ty_i:.1f}" x2="{tx_o:.1f}" y2="{ty_o:.1f}" '
            f'stroke="{tcol}" stroke-width="1.5" opacity="0.6"/>'
        )
        lx, ly = arc_point(cx, cy, r_outer + 16, tick_deg)
        svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="monospace" font-size="9" fill="{tcol}" opacity="0.7">{label}</text>'
        )

    # ── Needle ────────────────────────────────────────────────────────────────
    # Shadow
    svg_parts.append(
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" '
        f'stroke="rgba(0,0,0,0.5)" stroke-width="5" stroke-linecap="round"/>'
    )
    # Main needle
    svg_parts.append(
        f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" '
        f'stroke="white" stroke-width="2.5" stroke-linecap="round" filter="url(#glow)"/>'
    )
    # Needle tip dot
    svg_parts.append(
        f'<circle cx="{nx:.2f}" cy="{ny:.2f}" r="5" fill="{color}" filter="url(#glow)"/>'
    )

    # ── Hub ───────────────────────────────────────────────────────────────────
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="url(#hubGrad)" '
                     f'stroke="{color}" stroke-width="2" opacity="0.9"/>')
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{color}" filter="url(#glow)"/>')

    # ── Score text ────────────────────────────────────────────────────────────
    svg_parts.append(
        f'<text x="{cx}" y="{cy - 28}" text-anchor="middle" '
        f'font-family="monospace" font-size="46" font-weight="900" '
        f'fill="{color}" filter="url(#glow)" opacity="0.95" letter-spacing="-2">'
        f'{score}</text>'
    )
    svg_parts.append(
        f'<text x="{cx}" y="{cy - 8}" text-anchor="middle" '
        f'font-family="monospace" font-size="9.5" fill="#8B9EC7" letter-spacing="4" opacity="0.8">'
        f'CREDIT SCORE</text>'
    )

    # ── Percentage bar under score ────────────────────────────────────────────
    bar_w, bar_h = 80, 3
    bx = cx - bar_w // 2
    by = cy + 4
    svg_parts.append(
        f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bar_h}" '
        f'rx="1.5" fill="rgba(255,255,255,0.08)"/>'
    )
    svg_parts.append(
        f'<rect x="{bx}" y="{by}" width="{int(bar_w * pct)}" height="{bar_h}" '
        f'rx="1.5" fill="{color}" opacity="0.7"/>'
    )

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def render():
    st.markdown("""
    <div class="section-header">
        <div class="section-eyebrow">◈ INTELLIGENT CREDIT ASSESSMENT</div>
        <div class="section-title">AI Credit Scoring Engine</div>
        <div class="section-subtitle">Input applicant data to receive an instant AI-powered credit score with full explanation</div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#00D4FF;
                    letter-spacing:0.15em; margin-bottom:1rem;">APPLICANT INFORMATION</div>
        """, unsafe_allow_html=True)

        # ── Quick fill buttons ────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🎲 Random Profile", use_container_width=True):
                import random
                st.session_state.rand_seed    = random.randint(0, 9999)
                st.session_state.profile_type = "random"
                st.session_state.pop("score_triggered", None)
                st.session_state.pop("score_result", None)
        with c2:
            if st.button("✅ Strong Profile", use_container_width=True):
                st.session_state.rand_seed    = 42
                st.session_state.profile_type = "strong"
                st.session_state.pop("score_triggered", None)
                st.session_state.pop("score_result", None)
        with c3:
            if st.button("⚠️ Risky Profile", use_container_width=True):
                st.session_state.rand_seed    = 777
                st.session_state.profile_type = "risky"
                st.session_state.pop("score_triggered", None)
                st.session_state.pop("score_result", None)

        seed         = st.session_state.get("rand_seed", 1)
        profile_type = st.session_state.get("profile_type", "random")
        default_app  = generate_synthetic_applicant(seed=seed)

        # ── Override key features so Strong / Risky profiles are meaningful ──
        # EXT_SOURCE carries ~30% model weight — must be set explicitly
        if profile_type == "strong":
            default_app.update({
                "EXT_SOURCE_1":        0.82,
                "EXT_SOURCE_2":        0.79,
                "EXT_SOURCE_3":        0.85,
                "AMT_INCOME_TOTAL":    180000,
                "AMT_CREDIT":          300000,
                "AMT_ANNUITY":         12000,
                "YEARS_EMPLOYED":      10,
                "ON_TIME_PAYMENTS_PCT":98,
                "BUREAU_OVERDUE_DEBT": 0,
                "FLAG_OWN_REALTY":     "Y",
                "FLAG_OWN_CAR":        "Y",
                "NAME_EDUCATION_TYPE": "Higher education",
                "CODE_GENDER":         "F",
                "CNT_CHILDREN":        1,
                "BUREAU_RECORDS":      2,
            })
        elif profile_type == "risky":
            default_app.update({
                "EXT_SOURCE_1":        0.08,
                "EXT_SOURCE_2":        0.06,
                "EXT_SOURCE_3":        0.10,
                "AMT_INCOME_TOTAL":    22500,
                "AMT_CREDIT":          270000,
                "AMT_ANNUITY":         18000,
                "YEARS_EMPLOYED":      0,
                "ON_TIME_PAYMENTS_PCT":35,
                "BUREAU_OVERDUE_DEBT": 45000,
                "FLAG_OWN_REALTY":     "N",
                "FLAG_OWN_CAR":        "N",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "CODE_GENDER":         "M",
                "CNT_CHILDREN":        3,
                "BUREAU_RECORDS":      8,
            })

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["💼 Financial", "👤 Personal", "📱 Alternative Data"])

        with tab1:
            income = st.number_input(
                "Annual Income (₹)",
                min_value=10000, max_value=5000000,
                value=int(default_app["AMT_INCOME_TOTAL"]),
                step=5000, format="%d"
            )
            loan_amount = st.number_input(
                "Requested Loan Amount (₹)",
                min_value=10000, max_value=5000000,
                value=int(default_app["AMT_CREDIT"]),
                step=10000, format="%d"
            )
            loan_annuity = st.number_input(
                "Monthly Annuity (₹)",
                min_value=1000, max_value=200000,
                value=int(default_app["AMT_ANNUITY"]),
                step=1000, format="%d"
            )
            employment_type = st.selectbox(
                "Employment Type",
                ["Working", "Commercial associate", "State servant", "Self-employed", "Pensioner", "Student"],
                index=["Working", "Commercial associate", "State servant", "Self-employed", "Pensioner", "Student"].index(
                    default_app["NAME_INCOME_TYPE"]
                ) if default_app["NAME_INCOME_TYPE"] in ["Working", "Commercial associate", "State servant", "Self-employed", "Pensioner", "Student"] else 0
            )
            years_employed = st.slider("Years Employed", 0, 40, int(default_app["YEARS_EMPLOYED"]))
            on_time = st.slider("On-time Payment Rate (%)", 0, 100, int(default_app["ON_TIME_PAYMENTS_PCT"]))
            bureau_overdue = st.number_input(
                "Bureau Overdue Debt (₹)",
                min_value=0, max_value=500000,
                value=int(default_app["BUREAU_OVERDUE_DEBT"]),
                step=1000
            )

        with tab2:
            age = st.slider("Age", 18, 70, int(default_app["AGE_YEARS"]))
            gender = st.selectbox("Gender", ["F", "M"],
                                  index=0 if default_app["CODE_GENDER"] == "F" else 1)
            education = st.selectbox(
                "Education Level",
                ["Secondary / secondary special", "Higher education",
                 "Incomplete higher", "Lower secondary", "Academic degree"],
                index=["Secondary / secondary special", "Higher education",
                       "Incomplete higher", "Lower secondary", "Academic degree"].index(
                    default_app["NAME_EDUCATION_TYPE"]
                ) if default_app["NAME_EDUCATION_TYPE"] in ["Secondary / secondary special", "Higher education",
                       "Incomplete higher", "Lower secondary", "Academic degree"] else 0
            )
            housing = st.selectbox(
                "Housing Type",
                ["House / apartment", "With parents", "Municipal apartment",
                 "Rented apartment", "Office apartment", "Co-op apartment"],
                index=0
            )
            family_status = st.selectbox(
                "Family Status",
                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
                index=0
            )
            num_children = st.slider("Number of Children", 0, 10, int(default_app["CNT_CHILDREN"]))

        with tab3:
            st.markdown('<div style="font-size:0.8rem; color:#8B9EC7; margin-bottom:1rem;">These alternative signals replace traditional credit history for unbanked applicants.</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                has_mobile     = st.checkbox("📱 Mobile Registered",   value=bool(default_app["FLAG_MOBIL"]))
                has_email      = st.checkbox("📧 Email Verified",      value=bool(default_app["FLAG_EMAIL"]))
                has_phone      = st.checkbox("📞 Phone Verified",      value=bool(default_app.get("FLAG_PHONE", 1)))
                has_work_phone = st.checkbox("🏢 Work Phone",          value=bool(default_app.get("FLAG_WORK_PHONE", 1)))
            with col_b:
                owns_car       = st.checkbox("🚗 Owns Car",            value=default_app["FLAG_OWN_CAR"] == "Y")
                owns_realty    = st.checkbox("🏠 Owns Property",       value=default_app["FLAG_OWN_REALTY"] == "Y")
                doc_3          = st.checkbox("📄 Document 3 Provided", value=bool(default_app["FLAG_DOCUMENT_3"]))
                doc_6          = st.checkbox("📄 Document 6 Provided", value=bool(default_app["FLAG_DOCUMENT_6"]))

            st.markdown("**External Credit Scores (0-1)**")
            ext1 = st.slider("EXT_SOURCE_1 (Telecom/Utility)", 0.0, 1.0, float(default_app["EXT_SOURCE_1"]), 0.01)
            ext2 = st.slider("EXT_SOURCE_2 (Alt Bureau)",      0.0, 1.0, float(default_app["EXT_SOURCE_2"]), 0.01)
            ext3 = st.slider("EXT_SOURCE_3 (Behavioral)",      0.0, 1.0, float(default_app["EXT_SOURCE_3"]), 0.01)
            bureau_records = st.slider("Bureau Records Count", 0, 20, int(default_app["BUREAU_RECORDS"]))

        # Build applicant dict
        applicant = {
            "AMT_INCOME_TOTAL":     income,
            "AMT_CREDIT":           loan_amount,
            "AMT_ANNUITY":          loan_annuity,
            "NAME_INCOME_TYPE":     employment_type,
            "YEARS_EMPLOYED":       years_employed,
            "DAYS_EMPLOYED":        -years_employed * 365,
            "AGE_YEARS":            age,
            "DAYS_BIRTH":           -age * 365,
            "CODE_GENDER":          gender,
            "NAME_EDUCATION_TYPE":  education,
            "NAME_HOUSING_TYPE":    housing,
            "NAME_FAMILY_STATUS":   family_status,
            "CNT_CHILDREN":         num_children,
            "CNT_FAM_MEMBERS":      num_children + 2,
            "FLAG_OWN_CAR":         "Y" if owns_car else "N",
            "FLAG_OWN_REALTY":      "Y" if owns_realty else "N",
            "FLAG_MOBIL":           int(has_mobile),
            "FLAG_EMAIL":           int(has_email),
            "FLAG_PHONE":           int(has_phone),
            "FLAG_WORK_PHONE":      int(has_work_phone),
            "FLAG_DOCUMENT_3":      int(doc_3),
            "FLAG_DOCUMENT_6":      int(doc_6),
            "EXT_SOURCE_1":         ext1,
            "EXT_SOURCE_2":         ext2,
            "EXT_SOURCE_3":         ext3,
            "ON_TIME_PAYMENTS_PCT": on_time,
            "LATE_30_PAYMENTS":     max(0, int((100 - on_time) / 20)),
            "LATE_60_PAYMENTS":     max(0, int((100 - on_time) / 40)),
            "LATE_90_PAYMENTS":     max(0, int((100 - on_time) / 80)),
            "BUREAU_RECORDS":       bureau_records,
            "BUREAU_OVERDUE_DEBT":  bureau_overdue,
            "ANNUITY_INCOME_RATIO": loan_annuity / max(income, 1),
            "INCOME_CREDIT_RATIO":  income / max(loan_amount, 1),
            "CREDIT_INCOME_RATIO":  loan_amount / max(income, 1),
            "REGION_RATING_CLIENT": 2,
            "DAYS_CREDIT_AVG":      -365,
            "NAME_CONTRACT_TYPE":   "Cash loans",
        }
        st.session_state["current_applicant"] = applicant

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            score_btn = st.button("⚡ Score Now", type="primary", use_container_width=True)
        with col_btn2:
            agent_btn = st.button("🤖 Full Agent Analysis", use_container_width=True)

    # ── RIGHT COLUMN: Results ─────────────────────────────────────────────────
    with col_result:

        # Determine whether to show results
        if score_btn or agent_btn:
            # User just clicked — compute and store result
            result = compute_credit_score(applicant)
            st.session_state["score_result"] = result
            st.session_state["score_triggered"] = True

        elif st.session_state.get("score_triggered"):
            # User already scored once — keep result live as inputs change
            result = compute_credit_score(applicant)
            st.session_state["score_result"] = result

        else:
            # Page just loaded or Quick Fill pressed — show placeholder only
            result = None

        # ── Show results panel ────────────────────────────────────────────────
        if result is not None:

            # Gauge
            st.markdown(
                f'<div style="text-align:center; margin: 0.5rem 0 1.5rem 0;">'
                f'{render_score_gauge(result["score"], result["risk_color"])}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Risk tier badge
            tier_styles = {
                "EXCELLENT": ("rgba(0,255,136,0.12)",  "#00FF88", "rgba(0,255,136,0.3)"),
                "GOOD":      ("rgba(123,255,0,0.12)",   "#7BFF00", "rgba(123,255,0,0.3)"),
                "FAIR":      ("rgba(255,184,0,0.12)",   "#FFB800", "rgba(255,184,0,0.3)"),
                "POOR":      ("rgba(255,112,67,0.12)",  "#FF7043", "rgba(255,112,67,0.3)"),
                "VERY POOR": ("rgba(255,69,96,0.12)",   "#FF4560", "rgba(255,69,96,0.3)"),
            }
            tb_bg, tb_color, tb_border = tier_styles.get(
                result["risk_tier"],
                ("rgba(0,212,255,0.12)", "#00D4FF", "rgba(0,212,255,0.3)")
            )
            st.markdown(
                f'<div style="text-align:center;margin-bottom:1.5rem;">'
                f'<span style="display:inline-block;padding:6px 16px;border-radius:100px;'
                f'font-size:0.85rem;font-weight:700;letter-spacing:0.08em;'
                f'background:{tb_bg};color:{tb_color};border:1px solid {tb_border};">'
                f'{result["risk_tier"]}</span></div>',
                unsafe_allow_html=True
            )

            # Key metrics
            c1, c2 = st.columns(2)
            metrics = [
                ("Approval Probability", f"{result['approval_probability']:.0%}", result["risk_color"]),
                ("Default Risk",         f"{result['default_probability']:.1%}",   "#FF4560"),
                ("Recommended Rate",     result["recommended_rate"],                "#FFB800"),
                ("Max Loan Amount",      f"₹{result['max_loan_amount']:,.0f}",      "#00D4FF"),
            ]
            for i, (label, val, color) in enumerate(metrics):
                col = c1 if i % 2 == 0 else c2
                with col:
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:0.75rem;">'
                        f'<div class="metric-value" style="color:{color}; font-size:1.4rem;">{val}</div>'
                        f'<div class="metric-label">{label}</div>'
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # Factor breakdown
            st.markdown("""
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#00D4FF;
                        letter-spacing:0.15em; margin:1.25rem 0 0.75rem;">FACTOR BREAKDOWN</div>
            """, unsafe_allow_html=True)

            for factor_name, factor_data in result["factors"].items():
                pct        = min(100, max(0, factor_data["score"]))
                impact     = factor_data["impact"]
                impact_str = f"+{impact:.0f}" if impact >= 0 else f"{impact:.0f}"
                color      = "#00FF88" if impact >= 0 else "#FF4560"
                st.markdown(
                    f'<div class="factor-bar-wrapper">'
                    f'<div class="factor-label-row">'
                    f'<span class="factor-name">{factor_name}</span>'
                    f'<span style="font-family:\'Space Mono\',monospace; font-size:0.75rem; color:{color};">'
                    f"{impact_str} pts</span></div>"
                    f'<div class="factor-track">'
                    f'<div class="factor-fill" style="width:{pct}%;"></div>'
                    f"</div></div>",
                    unsafe_allow_html=True
                )

            # Quick AI insight — only fires on the button click, not on reruns
            if score_btn:
                st.markdown("---")
                with st.spinner("Getting AI quick assessment..."):
                    orch = CreditScoringOrchestrator()
                    insight = orch.quick_assess(applicant, result)
                st.markdown(
                    f'<div class="agent-message">'
                    f'<div class="agent-header">'
                    f'<span class="agent-icon">🤖</span>'
                    f'<span class="agent-name">AI Quick Assessment</span>'
                    f'<span class="agent-time">Just now</span>'
                    f"</div>"
                    f'<div class="agent-body">{insight}</div>'
                    f"</div>",
                    unsafe_allow_html=True
                )

            if agent_btn:
                st.session_state.current_page = "agentic_analysis"
                st.rerun()

        # ── Placeholder — shown before first Score click ───────────────────
        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                        height:400px; border:1px dashed rgba(0,212,255,0.2); border-radius:20px;
                        text-align:center; padding:2rem;">
                <div style="font-size:3rem; margin-bottom:1rem;">◈</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.2rem; color:#8B9EC7;">
                    Fill in applicant details and click Score Now
                </div>
                <div style="font-size:0.82rem; color:#4A5568; margin-top:0.5rem;">
                    Or use Quick Fill to load a sample profile
                </div>
            </div>
            """, unsafe_allow_html=True)
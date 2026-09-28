"""
AI-Powered Alternate Credit Scoring System
Main Streamlit Application Entry Point
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="CreditAI — Alternative Credit Scoring",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo",
        "About": "AI-Powered Alternative Credit Scoring System for Financial Inclusion"
    }
)

# Load custom CSS
from utils.styling import load_css
load_css()

# Import pages
from pages import (
    home,
    credit_assessment,
    agentic_analysis,
    data_explorer,
    model_insights,
    applicant_dashboard
)

# Sidebar Navigation
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <span class="logo-icon">◈</span>
            <span class="logo-text">CreditAI</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-tagline">Financial Inclusion Through Intelligence</div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        pages = {
            "🏠 Home": "home",
            "🧠 AI Credit Assessment": "credit_assessment",
            "🤖 Agentic Analysis": "agentic_analysis",
            "📊 Data Explorer": "data_explorer",
            "🔬 Model Insights": "model_insights",
            "👤 Applicant Dashboard": "applicant_dashboard",
        }

        if "current_page" not in st.session_state:
            st.session_state.current_page = "home"

        st.markdown('<div class="nav-label">NAVIGATION</div>', unsafe_allow_html=True)

        for label, page_key in pages.items():
            is_active = st.session_state.current_page == page_key
            btn_class = "nav-btn active" if is_active else "nav-btn"
            if st.button(label, key=f"nav_{page_key}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.current_page = page_key
                st.rerun()

        st.markdown("---")

        # API Configuration & Status
        st.markdown('<div class="nav-label">AI ENGINE CONFIGURATION</div>', unsafe_allow_html=True)
        from agents.credit_agents import (
            _get_gemini_api_key,
            _get_api_key as _get_groq_api_key,
            get_active_model
        )

        provider_choice = st.radio(
            "Provider",
            ["Google Gemini", "Groq"],
            index=0 if st.session_state.get("ai_provider", "gemini") == "gemini" else 1,
            horizontal=True,
            help="Select which AI engine powers the multi-agent reasoning."
        )
        selected_provider = "gemini" if "Gemini" in provider_choice else "groq"
        st.session_state.ai_provider = selected_provider

        gemini_key = _get_gemini_api_key()
        groq_key = _get_groq_api_key()
        active_has_key = bool(gemini_key) if selected_provider == "gemini" else bool(groq_key)

        with st.expander("🔑 AI Provider & API Settings", expanded=not active_has_key):
            if selected_provider == "gemini":
                st.markdown("**Google Gemini (Recommended)**")
                g_key = st.text_input(
                    "Gemini API Key",
                    value=st.session_state.get("gemini_api_key", gemini_key),
                    type="password",
                    placeholder="AIzaSy...",
                    help="Free API key from https://aistudio.google.com"
                )
                if g_key and g_key.strip() != gemini_key:
                    st.session_state.gemini_api_key = g_key.strip()
                    # Persist to secrets.toml
                    try:
                        import os
                        sec_dir = os.path.join(os.path.dirname(__file__), ".streamlit")
                        os.makedirs(sec_dir, exist_ok=True)
                        sec_file = os.path.join(sec_dir, "secrets.toml")
                        existing = ""
                        if os.path.exists(sec_file):
                            with open(sec_file, "r", encoding="utf-8") as f:
                                existing = f.read()
                        if "[gemini]" not in existing:
                            with open(sec_file, "a", encoding="utf-8") as f:
                                f.write(f'\n[gemini]\napi_key = "{g_key.strip()}"\n')
                        st.success("Gemini API key configured!")
                        st.rerun()
                    except Exception:
                        pass
                st.caption(f"Model: `{get_active_model()}`")
                st.caption("⚡ High speed & high rate limits")
            else:
                st.markdown("**Groq AI**")
                gr_key = st.text_input(
                    "Groq API Key",
                    value=st.session_state.get("groq_api_key", groq_key),
                    type="password",
                    placeholder="gsk_...",
                    help="Free API key from https://console.groq.com"
                )
                if gr_key and gr_key.strip() != groq_key:
                    st.session_state.groq_api_key = gr_key.strip()
                    st.success("Groq API key updated!")
                    st.rerun()
                st.caption(f"Model: `{get_active_model()}`")

        # System Status
        st.markdown('<div class="nav-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="status-grid">
            <div class="status-item">
                <span class="status-dot {'green' if active_has_key else 'yellow'}"></span>
                <span>AI Engine</span>
            </div>
            <div class="status-item">
                <span class="status-dot green"></span>
                <span>ML Models</span>
            </div>
            <div class="status-item">
                <span class="status-dot green"></span>
                <span>Data Pipeline</span>
            </div>
            <div class="status-item">
                <span class="status-dot {'green' if active_has_key else 'yellow'}"></span>
                <span>Agents</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-footer">
            <div>Powered by Claude AI</div>
            <div>Home Credit Dataset</div>
            <div>v2.0 — 2024</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    render_sidebar()

    page = st.session_state.get("current_page", "home")

    if page == "home":
        home.render()
    elif page == "credit_assessment":
        credit_assessment.render()
    elif page == "agentic_analysis":
        agentic_analysis.render()
    elif page == "data_explorer":
        data_explorer.render()
    elif page == "model_insights":
        model_insights.render()
    elif page == "applicant_dashboard":
        applicant_dashboard.render()


if __name__ == "__main__":
    main()

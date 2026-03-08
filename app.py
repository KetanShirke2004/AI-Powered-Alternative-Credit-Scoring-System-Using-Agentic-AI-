"""
AI-Powered Alternate Credit Scoring System
Main Streamlit Application Entry Point
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

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

        # System Status
        st.markdown('<div class="nav-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="status-grid">
            <div class="status-item">
                <span class="status-dot green"></span>
                <span>AI Engine</span>
            </div>
            <div class="status-item">
                <span class="status-dot green"></span>
                <span>ML Models</span>
            </div>
            <div class="status-item">
                <span class="status-dot yellow"></span>
                <span>Data Pipeline</span>
            </div>
            <div class="status-item">
                <span class="status-dot green"></span>
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

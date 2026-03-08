"""
Custom styling for CreditAI Streamlit application
Dark luxury fintech aesthetic with neon accents
"""

import streamlit as st


def load_css():
    css = """
    <style>
    /* ===== IMPORTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ===== ROOT VARIABLES ===== */
    :root {
        --bg-primary: #080C14;
        --bg-secondary: #0D1421;
        --bg-card: #111827;
        --bg-card-hover: #162032;
        --accent-cyan: #00D4FF;
        --accent-green: #00FF88;
        --accent-amber: #FFB800;
        --accent-red: #FF4560;
        --accent-purple: #7B61FF;
        --text-primary: #F0F4FF;
        --text-secondary: #8B9EC7;
        --text-muted: #4A5568;
        --border: rgba(0, 212, 255, 0.12);
        --border-hover: rgba(0, 212, 255, 0.35);
        --glow-cyan: 0 0 20px rgba(0, 212, 255, 0.3);
        --glow-green: 0 0 20px rgba(0, 255, 136, 0.3);
        --radius: 12px;
        --radius-lg: 20px;
    }

    /* ===== GLOBAL ===== */
    .stApp {
        background: var(--bg-primary) !important;
        background-image:
            radial-gradient(ellipse at 20% 10%, rgba(0, 212, 255, 0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(123, 97, 255, 0.04) 0%, transparent 50%) !important;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
        width: 260px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1rem !important;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.5rem 0.5rem 0.25rem;
        margin-bottom: 4px;
    }

    .logo-icon {
        font-size: 2rem;
        color: var(--accent-cyan);
        filter: drop-shadow(0 0 8px var(--accent-cyan));
        animation: pulse 3s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; filter: drop-shadow(0 0 8px var(--accent-cyan)); }
        50% { opacity: 0.8; filter: drop-shadow(0 0 16px var(--accent-cyan)); }
    }

    .logo-text {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }

    .sidebar-tagline {
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 0.05em;
        padding: 0 0.5rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }

    .nav-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        color: var(--text-muted);
        letter-spacing: 0.15em;
        padding: 0 0.5rem;
        margin: 0.5rem 0 0.4rem;
        text-transform: uppercase;
    }

    /* Nav buttons */
    [data-testid="stSidebar"] .stButton button {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem 0.75rem !important;
        margin: 1px 0 !important;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(0, 212, 255, 0.08) !important;
        border-color: var(--border-hover) !important;
        color: var(--accent-cyan) !important;
    }

    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: rgba(0, 212, 255, 0.12) !important;
        border-color: var(--accent-cyan) !important;
        color: var(--accent-cyan) !important;
    }

    .status-grid { padding: 0 0.25rem; }
    .status-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 4px;
        font-size: 0.78rem;
        color: var(--text-secondary);
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.green { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
    .status-dot.yellow { background: var(--accent-amber); box-shadow: 0 0 6px var(--accent-amber); }
    .status-dot.red { background: var(--accent-red); box-shadow: 0 0 6px var(--accent-red); }

    .sidebar-footer {
        font-size: 0.68rem;
        color: var(--text-muted);
        line-height: 2;
        padding: 0 0.5rem;
        font-family: 'Space Mono', monospace;
    }

    /* ===== MAIN CONTENT ===== */
    .main .block-container {
        padding: 2rem 2.5rem !important;
        max-width: 1400px !important;
    }

    /* ===== HEADINGS ===== */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
    }

    /* ===== CARDS ===== */
    .credit-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .credit-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }

    .credit-card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--glow-cyan);
    }

    .credit-card:hover::before { opacity: 1; }

    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.4rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* ===== SCORE RING ===== */
    .score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }

    .score-ring {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        position: relative;
        margin-bottom: 1rem;
    }

    .score-number {
        font-family: 'Space Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
    }

    .score-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* ===== BADGES ===== */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-success {
        background: rgba(0, 255, 136, 0.12);
        color: var(--accent-green);
        border: 1px solid rgba(0, 255, 136, 0.3);
    }

    .badge-warning {
        background: rgba(255, 184, 0, 0.12);
        color: var(--accent-amber);
        border: 1px solid rgba(255, 184, 0, 0.3);
    }

    .badge-danger {
        background: rgba(255, 69, 96, 0.12);
        color: var(--accent-red);
        border: 1px solid rgba(255, 69, 96, 0.3);
    }

    .badge-info {
        background: rgba(0, 212, 255, 0.12);
        color: var(--accent-cyan);
        border: 1px solid rgba(0, 212, 255, 0.3);
    }

    /* ===== AGENT CHAT ===== */
    .agent-message {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        position: relative;
    }

    .agent-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }

    .agent-icon {
        font-size: 1.1rem;
    }

    .agent-name {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: var(--accent-cyan);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .agent-time {
        font-size: 0.65rem;
        color: var(--text-muted);
        margin-left: auto;
    }

    .agent-body {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* ===== THINKING ANIMATION ===== */
    .thinking-dots {
        display: flex;
        gap: 4px;
        padding: 1rem;
        align-items: center;
    }

    .thinking-dots span {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent-cyan);
        animation: blink 1.4s ease-in-out infinite;
    }

    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes blink {
        0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
        40% { opacity: 1; transform: scale(1); }
    }

    /* ===== PROGRESS BAR ===== */
    .factor-bar-wrapper {
        margin-bottom: 0.75rem;
    }

    .factor-label-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        font-size: 0.8rem;
    }

    .factor-name { color: var(--text-secondary); }
    .factor-pct { font-family: 'Space Mono', monospace; color: var(--accent-cyan); }

    .factor-track {
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 100px;
        overflow: hidden;
    }

    .factor-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        margin-bottom: 1.5rem;
    }

    .section-eyebrow {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: var(--accent-cyan);
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.2;
        margin: 0;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        line-height: 1.6;
    }

    /* ===== FORMS ===== */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15) !important;
    }

    /* Slider */
    .stSlider .stSlider > div > div {
        background: var(--accent-cyan) !important;
    }

    /* ===== BUTTONS ===== */
    .main .stButton button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
        border: none !important;
        border-radius: 8px !important;
        color: #000 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.03em !important;
        transition: all 0.2s ease !important;
    }

    .main .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.4) !important;
    }

    .main .stButton button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--border-hover) !important;
        border-radius: 8px !important;
        color: var(--accent-cyan) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        border-radius: 0 !important;
        padding: 0.75rem 1.25rem !important;
        border-bottom: 2px solid transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom-color: var(--accent-cyan) !important;
        background: transparent !important;
    }

    /* ===== DATAFRAME ===== */
    .dataframe {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
    }

    /* ===== ALERTS ===== */
    .stAlert {
        background: var(--bg-card) !important;
        border-radius: var(--radius) !important;
    }

    /* ===== DIVIDER ===== */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.5rem 0 !important;
    }

    /* ===== HERO BANNER ===== */
    .hero-banner {
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,97,255,0.08));
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 3rem 2.5rem;
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
    }

    .hero-banner::after {
        content: '◈';
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 8rem;
        color: rgba(0, 212, 255, 0.06);
        font-family: 'Syne', sans-serif;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--text-primary) 50%, var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        max-width: 580px;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }

    /* ===== PIPELINE STEPS ===== */
    .pipeline-step {
        display: flex;
        gap: 1rem;
        padding: 1rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }

    .pipeline-step:hover {
        border-color: var(--border-hover);
    }

    .step-num {
        width: 32px; height: 32px;
        border-radius: 8px;
        background: rgba(0,212,255,0.12);
        border: 1px solid var(--border-hover);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent-cyan);
        flex-shrink: 0;
    }

    .step-content h4 {
        font-family: 'Syne', sans-serif;
        font-size: 0.9rem;
        font-weight: 700;
        margin: 0 0 2px;
        color: var(--text-primary);
    }

    .step-content p {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.5;
    }

    /* ===== RISK GAUGE ===== */
    .risk-gauge {
        padding: 1.5rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ===== SPINNER ===== */
    .stSpinner > div > div {
        border-top-color: var(--accent-cyan) !important;
    }

    /* ===== CUSTOM SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 212, 255, 0.3);
        border-radius: 100px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 212, 255, 0.6);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

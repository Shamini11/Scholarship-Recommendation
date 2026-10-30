"""
webapp.py - QuantumScholar AI
Premium, Modern, Minimal SaaS Application for AI Scholarship Recommendations & Eligibility Analysis

Palette System:
- Primary background: #F7F8FA
- Primary blue (accents): #52769B
- Deep navy (headings & navbar): #23364D
- Primary button blue: #355E85
- White cards: #FFFFFF
- Pale blue background: #E8EFF6
- Main text: #263238
- Secondary text: #687586
- Border colour: #E2E7ED
- Success badge: #E2F3E8 (text: #246B45)
- Warning badge: #FFF2D5 (text: #8A5A00)
- Error badge: #FCE8E8 (text: #A83232)
"""

import os
import io
import json
import urllib.parse
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components

from auth import (
    register_user,
    authenticate_user,
    load_users
)
from engine import (
    ScholarshipDataLoader,
    CumulativeProfileManager,
    GeminiNLPExtractor,
    ScholarshipScoringEngine,
    ScholarshipImprovementAdvisor,
    DEFAULT_GEMINI_API_KEY,
    DEFAULT_MODEL_NAME,
    EMPTY_PROFILE
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantumScholar AI — Scholarship Recommendation & Eligibility",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. PREMIUM LIGHT THEME STYLING (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Light Theme Reset */
    .stApp {
        background-color: #F7F8FA !important;
        color: #263238 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #23364D !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    p, span, label, div {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit header and footer clutter */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    footer {
        display: none !important;
    }

    /* Top Navigation Bar */
    .qs-navbar {
        background-color: #FFFFFF;
        border: 1px solid #E2E7ED;
        border-radius: 14px;
        padding: 0.85rem 1.4rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px 0 rgba(16, 24, 40, 0.04);
    }
    .qs-nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .qs-nav-icon {
        font-size: 1.35rem;
        color: #355E85;
        background: #E8EFF6;
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
    }
    .qs-nav-title {
        font-weight: 700;
        font-size: 1.2rem;
        color: #23364D;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .qs-nav-subtitle {
        font-size: 0.76rem;
        color: #687586;
        font-weight: 500;
    }
    .qs-nav-status {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .qs-status-pill {
        background-color: #E8EFF6;
        color: #355E85;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        border: 1px solid #D0DFEE;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .qs-user-pill {
        background-color: #F7F8FA;
        color: #23364D;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        border: 1px solid #E2E7ED;
    }

    /* Dashboard Header */
    .qs-page-header {
        margin-bottom: 1.25rem;
    }
    .qs-page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #23364D;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.4px;
    }
    .qs-page-subtitle {
        font-size: 0.95rem;
        color: #687586;
        margin: 0;
        line-height: 1.5;
    }

    /* Section Headings */
    .qs-section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #23364D;
        letter-spacing: -0.2px;
        margin-bottom: 0.25rem;
    }
    .qs-section-desc {
        font-size: 0.86rem;
        color: #687586;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    /* White Cards */
    .qs-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E7ED !important;
        border-radius: 14px !important;
        padding: 1.35rem 1.45rem !important;
        margin-bottom: 1.25rem !important;
        box-shadow: 0 1px 3px 0 rgba(16, 24, 40, 0.04) !important;
    }
    .qs-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #23364D;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .qs-card-desc {
        font-size: 0.84rem;
        color: #687586;
        margin-bottom: 0.9rem;
        line-height: 1.45;
    }

    /* Form Controls & Inputs */
    input[type="text"], input[type="password"], textarea, .stTextInput input, .stTextArea textarea {
        color: #263238 !important;
        background-color: #FFFFFF !important;
        font-size: 0.92rem !important;
        font-weight: 400 !important;
        border: 1px solid #E2E7ED !important;
        border-radius: 8px !important;
        padding: 0.65rem 0.85rem !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.02) !important;
        font-family: 'Inter', sans-serif !important;
    }
    input[type="text"]:focus, input[type="password"]:focus, textarea:focus, .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #52769B !important;
        box-shadow: 0 0 0 3px rgba(82, 118, 155, 0.15) !important;
        outline: none !important;
    }
    ::placeholder {
        color: #94A3B8 !important;
    }

    /* Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E7ED !important;
        border-radius: 8px !important;
        color: #263238 !important;
    }

    /* Buttons */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"],
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] button,
    button[data-testid="baseButton-primaryFormSubmit"],
    .stLinkButton a {
        background-color: #355E85 !important;
        background: #355E85 !important;
        color: #FFFFFF !important;
        border: 1px solid #355E85 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05) !important;
        transition: all 0.15s ease-in-out !important;
        text-decoration: none !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    button[data-testid="baseButton-primaryFormSubmit"]:hover,
    .stLinkButton a:hover {
        background-color: #2B4E6F !important;
        background: #2B4E6F !important;
        border-color: #2B4E6F !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(16, 24, 40, 0.08) !important;
    }
    div.stButton > button[kind="secondary"], div.stButton > button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        color: #355E85 !important;
        border: 1px solid #E2E7ED !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.1rem !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button[kind="secondary"]:hover, div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #F7F8FA !important;
        border-color: #52769B !important;
        color: #23364D !important;
    }

    /* Tabs Styling */
    [data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 8px !important;
        border-bottom: 1px solid #E2E7ED !important;
    }
    [data-baseweb="tab"] {
        color: #687586 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
    }
    [aria-selected="true"] {
        color: #23364D !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #355E85 !important;
    }

    /* Profile Attributes Display */
    .qs-chip-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.65rem 0 0.85rem 0;
    }
    .qs-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background-color: #F7F8FA;
        color: #263238;
        border: 1px solid #E2E7ED;
        padding: 0.35rem 0.65rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .qs-chip b {
        color: #23364D;
        font-weight: 600;
    }
    .qs-empty-profile {
        background-color: #F7F8FA;
        border: 1px dashed #D0DFEE;
        border-radius: 10px;
        padding: 1.5rem 1rem;
        text-align: center;
        color: #687586;
        font-size: 0.88rem;
        margin-bottom: 0.75rem;
    }

    /* Scholarship Recommendation Cards */
    .qs-scholarship-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E7ED !important;
        border-left: 4px solid #52769B !important;
        border-radius: 14px !important;
        padding: 1.35rem 1.5rem !important;
        margin-bottom: 1.25rem !important;
        box-shadow: 0 1px 3px 0 rgba(16, 24, 40, 0.04) !important;
        transition: box-shadow 0.15s ease-in-out, transform 0.15s ease-in-out !important;
    }
    .qs-scholarship-card:hover {
        box-shadow: 0 4px 12px 0 rgba(16, 24, 40, 0.08) !important;
        transform: translateY(-2px) !important;
    }
    .qs-card-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.85rem;
        margin-bottom: 0.65rem;
    }
    .qs-scheme-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #23364D;
        line-height: 1.35;
        letter-spacing: -0.2px;
    }
    .qs-provider-tag {
        background-color: #E8EFF6;
        color: #355E85;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    .qs-badge-eligible {
        background-color: #E2F3E8 !important;
        color: #246B45 !important;
        border: 1px solid #C3E6D0 !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 9999px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .qs-badge-potential {
        background-color: #FFF2D5 !important;
        color: #8A5A00 !important;
        border: 1px solid #FFE0A3 !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 9999px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .qs-score-pill {
        background-color: #F7F8FA;
        color: #355E85;
        border: 1px solid #E2E7ED;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    .qs-benefit-panel {
        background-color: #F7F8FA;
        border: 1px solid #E2E7ED;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        font-size: 0.88rem;
        color: #23364D;
        font-weight: 600;
        margin: 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .qs-reason-pill {
        display: inline-block;
        background-color: #E8EFF6;
        color: #23364D;
        border: 1px solid #D0DFEE;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    /* Pale Blue Empty State */
    .qs-empty-state {
        background-color: #E8EFF6;
        border: 1px solid #D0DFEE;
        border-radius: 14px;
        padding: 2.8rem 1.5rem;
        text-align: center;
        margin-bottom: 1.25rem;
    }
    .qs-empty-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #23364D;
        margin-bottom: 0.35rem;
    }
    .qs-empty-sub {
        font-size: 0.88rem;
        color: #687586;
        max-width: 440px;
        margin: 0 auto;
        line-height: 1.45;
    }

    /* Profile Improvement Tip Card */
    .qs-tip-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E7ED !important;
        border-left: 3px solid #52769B !important;
        border-radius: 10px !important;
        padding: 0.95rem 1.15rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03) !important;
    }
    .qs-tip-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #23364D;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .qs-tip-badge {
        background-color: #E8EFF6;
        color: #355E85;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
    }
    .qs-tip-desc {
        font-size: 0.84rem;
        color: #687586;
        line-height: 1.45;
        margin-top: 0.3rem;
    }

    /* Chat Feed Light Theme */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E7ED !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.65rem !important;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.02) !important;
        color: #263238 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E7ED !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.03) !important;
        margin-bottom: 1rem !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #23364D !important;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div > div > div {
        background-color: #52769B !important;
    }
    [data-testid="stProgress"] > div > div {
        background-color: #E2E7ED !important;
        border-radius: 6px !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #23364D !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #687586 !important;
        font-size: 0.82rem !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. CACHED DATASET INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_data
def get_verified_dataset():
    loader = ScholarshipDataLoader()
    return loader.df

try:
    df_dataset = get_verified_dataset()
except Exception as e:
    st.error(f"Error loading scholarship dataset: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# 4. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = CumulativeProfileManager.get_initial_profile()

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = DEFAULT_GEMINI_API_KEY

if 'model_name' not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL_NAME

if 'active_tab_right' not in st.session_state:
    st.session_state.active_tab_right = "matches"

if 'input_key_counter' not in st.session_state:
    st.session_state.input_key_counter = 0

if 'voice_prefill' not in st.session_state:
    st.session_state.voice_prefill = ""

if 'voice_toast_msg' not in st.session_state:
    st.session_state.voice_toast_msg = None


# =============================================================================
# 5. HANDLE INCOMING VOICE DICTATION VIA QUERY PARAMETER (AUTO-FILL INPUT)
# =============================================================================
if "voice_transcript" in st.query_params or "voice_prompt" in st.query_params:
    raw_voice_val = st.query_params.get("voice_transcript") or st.query_params.get("voice_prompt", "")
    st.query_params.clear()
    
    if raw_voice_val and str(raw_voice_val).strip():
        text_received = urllib.parse.unquote(str(raw_voice_val)).strip()
        if text_received:
            # Prefill speech into text input and advance counter for a fresh bound widget
            st.session_state.voice_prefill = text_received
            st.session_state.voice_toast_msg = text_received
            st.session_state.input_key_counter += 1
            st.rerun()


# =============================================================================
# 6. AUTHENTICATION VIEW (DESKTOP TWO-COLUMN REDESIGN)
# =============================================================================
if not st.session_state.authenticated:
    col_auth_left, col_auth_right = st.columns([1.1, 1], gap="large")

    with col_auth_left:
        st.markdown("""
        <div style="padding: 1.5rem 0.5rem 1.5rem 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem;">
                <span class="qs-nav-icon">🎓</span>
                <div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #23364D; letter-spacing: -0.3px;">QuantumScholar AI</div>
                    <div style="font-size: 0.78rem; color: #687586; font-weight: 500;">National & State Scholarship Intelligence Platform</div>
                </div>
            </div>
            <h1 style="font-size: 2.1rem; font-weight: 700; color: #23364D; line-height: 1.25; margin: 0 0 0.85rem 0; letter-spacing: -0.5px;">
                Find the Scholarships You Deserve.
            </h1>
            <p style="font-size: 1rem; color: #687586; line-height: 1.55; margin-bottom: 1.75rem;">
                Discover scholarship opportunities tailored to your academic background, degree level, and community eligibility.
            </p>
            <div style="display: flex; flex-direction: column; gap: 0.85rem;">
                <div style="background: #FFFFFF; border: 1px solid #E2E7ED; border-left: 3px solid #52769B; border-radius: 10px; padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(16,24,40,0.03);">
                    <div style="font-weight: 600; font-size: 0.92rem; color: #23364D;">🎯 Personalised Scholarship Recommendations</div>
                    <div style="font-size: 0.82rem; color: #687586; margin-top: 0.2rem;">Evaluates 72+ verified schemes against your marks, income ceiling, and community criteria.</div>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E2E7ED; border-left: 3px solid #52769B; border-radius: 10px; padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(16,24,40,0.03);">
                    <div style="font-weight: 600; font-size: 0.92rem; color: #23364D;">📋 Eligibility Analysis</div>
                    <div style="font-size: 0.82rem; color: #687586; margin-top: 0.2rem;">Clear criteria breakdown explaining why you qualify with official documentation requirements.</div>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E2E7ED; border-left: 3px solid #52769B; border-radius: 10px; padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(16,24,40,0.03);">
                    <div style="font-weight: 600; font-size: 0.92rem; color: #23364D;">🏛️ Tamil Nadu & National Opportunities</div>
                    <div style="font-size: 0.82rem; color: #687586; margin-top: 0.2rem;">Comprehensive coverage of state welfare schemes (BC/MBC/SC/ST, Pudhumai Penn) & central portals.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_auth_right:
        with st.container():
            st.markdown('<div class="qs-card" style="box-shadow: 0 4px 16px rgba(16, 24, 40, 0.06); padding: 1.75rem 1.85rem; margin-top: 1rem;">', unsafe_allow_html=True)
            auth_in_tab, auth_up_tab = st.tabs(["🔑 Sign In", "📝 Create Account"])

            with auth_in_tab:
                st.markdown("<div style='font-size: 1.25rem; font-weight: 700; color: #23364D; margin-bottom: 0.2rem;'>Welcome Back</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 0.85rem; color: #687586; margin-bottom: 1rem;'>Sign in with your registered credentials to access your dashboard.</div>", unsafe_allow_html=True)

                with st.form("qs_signin_form"):
                    in_email = st.text_input("Email Address", placeholder="student@example.com").strip()
                    in_pass = st.text_input("Password", type="password", placeholder="••••••••")
                    submit_in = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                    if submit_in:
                        ok, user_data, msg = authenticate_user(in_email, in_pass)
                        if ok:
                            st.session_state.authenticated = True
                            st.session_state.current_user = user_data
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

                st.markdown("<div style='margin-top: 0.9rem; padding: 0.75rem 0.9rem; background: #F7F8FA; border: 1px solid #E2E7ED; border-radius: 8px; font-size: 0.82rem; color: #687586;'>💡 <b>Demo Account:</b> <code>student@example.com</code> / <code>Password@123</code></div>", unsafe_allow_html=True)
                if st.button("Use Demo Account & Sign In", type="secondary", use_container_width=True):
                    ok, user_data, _ = authenticate_user("student@example.com", "Password@123")
                    if ok:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user_data
                        st.rerun()

            with auth_up_tab:
                st.markdown("<div style='font-size: 1.25rem; font-weight: 700; color: #23364D; margin-bottom: 0.2rem;'>Create Your Account</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size: 0.85rem; color: #687586; margin-bottom: 1rem;'>Register a new profile to receive tailored scholarship recommendations.</div>", unsafe_allow_html=True)

                with st.form("qs_signup_form"):
                    up_name = st.text_input("Full Name", placeholder="Shamini K").strip()
                    up_phone = st.text_input("Phone Number", placeholder="9876543210").strip()
                    up_email = st.text_input("Email ID", placeholder="yourname@gmail.com").strip()
                    up_pass = st.text_input("Create Password", type="password", placeholder="Min 6 characters")
                    up_conf = st.text_input("Confirm Password", type="password", placeholder="Re-type password")
                    submit_up = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                    if submit_up:
                        ok, msg = register_user(up_name, up_phone, up_email, up_pass, up_conf)
                        if ok:
                            auth_ok, user_data, _ = authenticate_user(up_email, up_pass)
                            if auth_ok:
                                st.session_state.authenticated = True
                                st.session_state.current_user = user_data
                                st.rerun()
                        else:
                            st.error(f"❌ {msg}")

            st.markdown('</div>', unsafe_allow_html=True)

    # Stop: Guard keeps main dashboard locked until signed in
    st.stop()


# =============================================================================
# 7. LOGGED-IN HOME PAGE: MODERN SAAS DASHBOARD
# =============================================================================

user = st.session_state.current_user or {"name": "Student", "email": "User", "phone": ""}
user_first_name = user.get("name", "Student").split()[0]
prof = st.session_state.user_profile

# Check if user has provided profile attributes
has_profile_data = any(
    prof.get(k) is not None and str(prof.get(k)).strip() != "" and str(prof.get(k)).strip() != "None"
    for k in ['community', 'gender', 'education_level', 'annual_income', 'min_marks_or_cgpa', 'disability', 'state', 'government_school', 'first_graduate']
)

# --- Top Navigation Bar ---
st.markdown(f"""
<div class="qs-navbar">
    <div class="qs-nav-brand">
        <span class="qs-nav-icon">🎓</span>
        <div>
            <div class="qs-nav-title">QuantumScholar AI</div>
            <div class="qs-nav-subtitle">Scholarship Recommendation & Eligibility Platform</div>
        </div>
    </div>
    <div class="qs-nav-status">
        <span class="qs-status-pill">● 72 Schemes Indexed & Verified</span>
        <span class="qs-user-pill">👤 {user['name']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Dashboard Title & Sign Out Bar
col_dash_title, col_dash_act = st.columns([5, 1])
with col_dash_title:
    st.markdown("""
    <div class="qs-page-header">
        <h1 class="qs-page-title">Your Scholarship Dashboard</h1>
        <p class="qs-page-subtitle">Discover funding opportunities matched to your academic profile.</p>
    </div>
    """, unsafe_allow_html=True)
with col_dash_act:
    if st.button("Sign Out 🚪", type="secondary", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.user_profile = CumulativeProfileManager.get_initial_profile()
        st.session_state.chat_messages = []
        st.session_state.voice_prefill = ""
        st.session_state.input_key_counter += 1
        st.session_state.voice_toast_msg = None
        st.session_state.active_tab_right = "matches"
        st.rerun()

# Initial Introductory AI Message
if not st.session_state.chat_messages:
    welcome_text = (
        "Welcome to QuantumScholar AI. Tell me about yourself, your academic background, "
        "community, income, and any specific requirements to find matching scholarships."
    )
    st.session_state.chat_messages.append({"role": "assistant", "content": welcome_text})


# =============================================================================
# SPLIT-SCREEN DASHBOARD: LEFT (INPUT & PROFILE) | RIGHT (RESULTS & TOOLS)
# =============================================================================
col_left, col_right = st.columns([1, 1.25], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN: CONVERSATIONAL AI & ACTIVE PROFILE SUMMARY
# -----------------------------------------------------------------------------
with col_left:
    st.markdown('<div class="qs-card">', unsafe_allow_html=True)
    st.markdown('<div class="qs-card-title">💬 AI Scholarship Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="qs-card-desc">Describe your academic background in natural language or use voice dictation.</div>', unsafe_allow_html=True)

    # Chat Messages Feed
    with st.container(height=260):
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    # Voice Dictation Toast Notification
    if st.session_state.get("voice_toast_msg"):
        st.markdown(f"""
        <div style="background-color: #E2F3E8; border: 1px solid #C3E6D0; border-radius: 8px; padding: 0.55rem 0.85rem; margin-bottom: 0.65rem; font-size: 0.84rem; color: #246B45; display: flex; align-items: center; gap: 0.45rem;">
            <span>✓</span>
            <span><b>Voice Captured & Auto-Filled:</b> "{st.session_state.voice_toast_msg}"</span>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.voice_toast_msg = None

    # Modern Web Speech API Voice Dictation Component
    voice_component_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: transparent;
            color: #263238;
            overflow: hidden;
        }
        .voice-card {
            background: #FFFFFF;
            border: 1px solid #E2E7ED;
            border-radius: 9px;
            padding: 8px 12px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .voice-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
        .voice-action-btn {
            background-color: #FFFFFF;
            color: #355E85;
            border: 1.5px solid #52769B;
            border-radius: 7px;
            padding: 6px 14px;
            font-family: 'Inter', sans-serif;
            font-size: 0.84rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease-in-out;
            white-space: nowrap;
        }
        .voice-action-btn:hover {
            background-color: #F7F8FA;
            border-color: #355E85;
            color: #23364D;
        }
        .voice-action-btn.recording {
            background-color: #FCE8E8 !important;
            border-color: #A83232 !important;
            color: #A83232 !important;
            box-shadow: 0 0 0 3px rgba(168, 50, 50, 0.12);
        }
        .voice-action-btn.processing {
            background-color: #FFF2D5 !important;
            border-color: #8A5A00 !important;
            color: #8A5A00 !important;
        }
        .status-pill {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.81rem;
            color: #687586;
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #94A3B8;
            display: inline-block;
            flex-shrink: 0;
            transition: all 0.2s ease;
        }
        .pulse-dot.listening {
            background-color: #A83232;
            box-shadow: 0 0 6px rgba(168, 50, 50, 0.7);
            animation: pulseWave 1.1s infinite ease-in-out;
        }
        .pulse-dot.processing {
            background-color: #8A5A00;
            animation: pulseWave 0.7s infinite ease-in-out;
        }
        .pulse-dot.success {
            background-color: #246B45;
            box-shadow: 0 0 6px rgba(36, 107, 69, 0.5);
        }
        @keyframes pulseWave {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.35); opacity: 0.4; }
        }
        .live-preview {
            font-size: 0.80rem;
            color: #23364D;
            background-color: #F7F8FA;
            border: 1px dashed #D0DFEE;
            border-radius: 6px;
            padding: 4px 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-style: italic;
        }
        .live-badge {
            font-style: normal;
            font-weight: 700;
            color: #355E85;
            font-size: 0.73rem;
            margin-right: 4px;
        }
        .error-msg {
            font-size: 0.78rem;
            color: #A83232;
            background-color: #FCE8E8;
            border-radius: 5px;
            padding: 4px 8px;
        }
    </style>
    </head>
    <body>
    <div class="voice-card">
        <div class="voice-bar">
            <button id="q-mic-btn" class="voice-action-btn" onclick="toggleDictation()">
                <span id="q-mic-icon">🎙️</span>
                <span id="q-mic-txt">Start Voice Dictation</span>
            </button>
            <div class="status-pill">
                <span id="q-pulse-dot" class="pulse-dot"></span>
                <span id="q-status-msg">Click microphone to speak (English)</span>
            </div>
        </div>
        <div id="q-preview-box" class="live-preview" style="display: none;">
            <span class="live-badge">LIVE TRANSCRIPT:</span>
            <span id="q-preview-txt">Listening...</span>
        </div>
        <div id="q-err-banner" class="error-msg" style="display: none;"></div>
    </div>

    <script>
        let recognizer = null;
        let isRecording = false;
        let finalOutput = '';

        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (SpeechRec) {
            recognizer = new SpeechRec();
            // Requirement 1: accurate local configuration
            recognizer.continuous = false;      // Concludes cleanly after speaking
            recognizer.interimResults = true;   // Live transcription feedback
            recognizer.lang = 'en-IN';          // Optimized for Indian English & academic terms
            recognizer.maxAlternatives = 1;

            recognizer.onstart = function() {
                isRecording = true;
                finalOutput = '';
                document.getElementById('q-err-banner').style.display = 'none';
                document.getElementById('q-preview-box').style.display = 'block';
                document.getElementById('q-preview-txt').innerText = 'Listening for speech...';

                const btn = document.getElementById('q-mic-btn');
                btn.className = 'voice-action-btn recording';
                document.getElementById('q-mic-icon').innerText = '⏹️';
                document.getElementById('q-mic-txt').innerText = 'Stop & Auto-Fill';

                const dot = document.getElementById('q-pulse-dot');
                dot.className = 'pulse-dot listening';
                document.getElementById('q-status-msg').innerHTML = '<span style="color: #A83232; font-weight: 600;">🎙️ Listening... Speak now</span>';
            };

            recognizer.onresult = function(event) {
                let interim = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    const item = event.results[i];
                    if (item.isFinal) {
                        finalOutput += item[0].transcript + ' ';
                    } else {
                        interim += item[0].transcript;
                    }
                }
                const liveText = (finalOutput + interim).trim();
                if (liveText.length > 0) {
                    document.getElementById('q-preview-txt').innerText = '"' + liveText + '"';
                    // Requirement 2: Immediate DOM auto-fill into parent Streamlit textarea
                    injectIntoParentTextarea(liveText);
                }
            };

            recognizer.onerror = function(event) {
                isRecording = false;
                resetButtonState();

                let message = 'Microphone notice: ' + event.error;
                if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                    message = '⚠️ Microphone permission denied. Please allow microphone access in your browser address bar.';
                } else if (event.error === 'no-speech') {
                    message = 'ℹ️ No speech was heard. Click the button and speak into your microphone.';
                } else if (event.error === 'network') {
                    message = '⚠️ Speech recognition network connection notice. Please retry.';
                }

                const errEl = document.getElementById('q-err-banner');
                errEl.innerText = message;
                errEl.style.display = 'block';
                document.getElementById('q-status-msg').innerText = 'Microphone ready';
            };

            recognizer.onend = function() {
                isRecording = false;
                const capturedText = finalOutput.trim();

                if (capturedText.length > 0) {
                    // Update UI to success state
                    const btn = document.getElementById('q-mic-btn');
                    btn.className = 'voice-action-btn processing';
                    document.getElementById('q-mic-icon').innerText = '⏳';
                    document.getElementById('q-mic-txt').innerText = 'Auto-Filling...';

                    const dot = document.getElementById('q-pulse-dot');
                    dot.className = 'pulse-dot success';
                    document.getElementById('q-status-msg').innerHTML = '<span style="color: #246B45; font-weight: 600;">✓ Transcript captured!</span>';

                    // 1. Direct DOM injection into Streamlit's textarea
                    injectIntoParentTextarea(capturedText);

                    // 2. Seamless Streamlit Rerun via query parameter to update session state
                    setTimeout(function() {
                        try {
                            const parentUrl = new URL(window.parent.location.href);
                            parentUrl.searchParams.set('voice_transcript', encodeURIComponent(capturedText));
                            window.parent.location.href = parentUrl.toString();
                        } catch(e) {
                            console.error('State sync error:', e);
                            resetButtonState();
                        }
                    }, 350);
                } else {
                    resetButtonState();
                }
            };
        } else {
            const btn = document.getElementById('q-mic-btn');
            btn.disabled = true;
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
            const errEl = document.getElementById('q-err-banner');
            errEl.innerText = '⚠️ Web Speech API is not supported in this browser. Please use Chrome, Edge, or Safari.';
            errEl.style.display = 'block';
        }

        function toggleDictation() {
            if (!recognizer) return;
            if (!isRecording) {
                try {
                    recognizer.start();
                } catch(e) {
                    console.warn(e);
                }
            } else {
                isRecording = false;
                try {
                    recognizer.stop();
                } catch(e) {
                    console.warn(e);
                }
            }
        }

        function resetButtonState() {
            isRecording = false;
            const btn = document.getElementById('q-mic-btn');
            btn.className = 'voice-action-btn';
            document.getElementById('q-mic-icon').innerText = '🎙️';
            document.getElementById('q-mic-txt').innerText = 'Start Voice Dictation';

            const dot = document.getElementById('q-pulse-dot');
            dot.className = 'pulse-dot';
            document.getElementById('q-status-msg').innerText = 'Click microphone to speak';
        }

        function injectIntoParentTextarea(text) {
            try {
                const parentDoc = window.parent.document;
                const textareas = parentDoc.querySelectorAll('textarea');
                textareas.forEach(function(ta) {
                    const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    if (nativeSetter) {
                        nativeSetter.call(ta, text);
                    } else {
                        ta.value = text;
                    }
                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                    ta.dispatchEvent(new Event('change', { bubbles: true }));
                });
            } catch(err) {
                console.warn('DOM auto-fill notice:', err);
            }
        }
    </script>
    </body>
    </html>
    """
    components.html(voice_component_html, height=72)

    # Text Input Form (with automatic session state binding)
    input_widget_key = f"student_input_{st.session_state.input_key_counter}"
    with st.form("student_input_form", clear_on_submit=False):
        user_text_input = st.text_area(
            label="Share your background",
            value=st.session_state.voice_prefill,
            placeholder="I'm an undergraduate student in Tamil Nadu. Here are my academic details, family income, and other relevant information...",
            height=95,
            label_visibility="collapsed",
            key=input_widget_key
        )
        st.markdown("<div style='font-size: 0.82rem; color: #687586; margin-bottom: 0.75rem;'>💡 Include your degree (UG/PG), community (BC/MBC/SC/ST/General), annual family income, marks or CGPA.</div>", unsafe_allow_html=True)
        send_clicked = st.form_submit_button("Find My Scholarships", type="primary", use_container_width=True)

        if send_clicked and user_text_input.strip():
            user_msg = user_text_input.strip()
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})

            with st.spinner("Analyzing your profile and matching scholarship criteria..."):
                extractor = GeminiNLPExtractor(
                    api_key=st.session_state.gemini_api_key,
                    model_name=st.session_state.model_name
                )
                extracted_dict, status_msg, ai_reply = extractor.extract_with_gemini(
                    text=user_msg,
                    current_profile=st.session_state.user_profile,
                    user_name=user['name'],
                    timeout_sec=1.0
                )

                updated_prof, _ = CumulativeProfileManager.merge(
                    current_profile=st.session_state.user_profile,
                    new_attributes=extracted_dict
                )
                st.session_state.user_profile = updated_prof
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_reply})
                st.session_state.active_tab_right = "matches"
                
                # Clear prefill and advance counter for fresh empty text area on next run
                st.session_state.voice_prefill = ""
                st.session_state.input_key_counter += 1
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Student Profile Summary Card
    st.markdown('<div class="qs-card">', unsafe_allow_html=True)
    st.markdown('<div class="qs-card-title">📋 Active Student Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="qs-card-desc">Extracted attributes calibrated from your conversation.</div>', unsafe_allow_html=True)

    profile_badge_items = [
        ("State", prof.get("state")),
        ("Community", prof.get("community")),
        ("Gender", prof.get("gender")),
        ("Education", prof.get("education_level")),
        ("Annual Income", f"₹{prof['annual_income']:,.0f}" if prof.get("annual_income") else None),
        ("Disability Status", prof.get("disability")),
        ("Marks / CGPA", f"{prof['min_marks_or_cgpa']}%" if prof.get("min_marks_or_cgpa") and prof['min_marks_or_cgpa'] > 10 else (f"{prof['min_marks_or_cgpa']} CGPA" if prof.get("min_marks_or_cgpa") else None)),
        ("Govt School", "Yes (7.5% Quota)" if prof.get("government_school") == "Yes" else ("No" if prof.get("government_school") == "No" else None)),
        ("First Graduate", "Yes" if prof.get("first_graduate") == "Yes" else ("No" if prof.get("first_graduate") == "No" else None)),
        ("Accommodation", prof.get("hosteller"))
    ]

    active_badges = [(lbl, val) for lbl, val in profile_badge_items if val and str(val).strip() != "" and str(val).strip() != "None"]

    if not active_badges:
        st.markdown("""
        <div class="qs-empty-profile">
            <div style="font-weight: 600; color: #23364D; margin-bottom: 0.25rem;">Your profile will appear here.</div>
            <div>Enter your academic details to build your scholarship profile.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        badges_html = "<div class='qs-chip-grid'>" + "".join([f"<span class='qs-chip'>{lbl}: <b>{val}</b></span>" for lbl, val in active_badges]) + "</div>"
        st.markdown(badges_html, unsafe_allow_html=True)

    completion_score = CumulativeProfileManager.calculate_completion_percentage(prof)
    st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #687586; margin-top: 0.75rem; margin-bottom: 0.35rem;'><span>Profile Calibration</span><span style='font-weight: 700; color: #355E85;'>{completion_score}%</span></div>", unsafe_allow_html=True)
    st.progress(completion_score / 100)

    st.markdown("<div style='height: 1px; background: #E2E7ED; margin: 1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #687586; margin-bottom: 0.45rem;'>Quick Simulation Triggers:</div>", unsafe_allow_html=True)
    col_qp1, col_qp2 = st.columns(2)
    with col_qp1:
        if st.button("Sample: Female BC UG (85%)", type="secondary", use_container_width=True):
            st.session_state.voice_prefill = ""
            st.session_state.input_key_counter += 1
            st.session_state.chat_messages.append({"role": "user", "content": "I am a female BC student studying UG in Tamil Nadu with 85% marks."})
            ext = GeminiNLPExtractor().extract_with_gemini("I am a female BC student studying UG in Tamil Nadu with 85% marks.", prof, user['name'])
            st.session_state.user_profile, _ = CumulativeProfileManager.merge(prof, ext[0])
            st.session_state.chat_messages.append({"role": "assistant", "content": ext[2]})
            st.session_state.active_tab_right = "matches"
            st.rerun()
    with col_qp2:
        if st.button("Sample: Differently Abled", type="secondary", use_container_width=True):
            st.session_state.voice_prefill = ""
            st.session_state.input_key_counter += 1
            st.session_state.chat_messages.append({"role": "user", "content": "I am a UG student in Tamil Nadu with a walking disability."})
            ext = GeminiNLPExtractor().extract_with_gemini("I am a UG student in Tamil Nadu with a walking disability.", prof, user['name'])
            st.session_state.user_profile, _ = CumulativeProfileManager.merge(prof, ext[0])
            st.session_state.chat_messages.append({"role": "assistant", "content": ext[2]})
            st.session_state.active_tab_right = "matches"
            st.rerun()

    if st.button("🔄 Reset Profile State", type="secondary", use_container_width=True):
        st.session_state.user_profile = CumulativeProfileManager.get_initial_profile()
        st.session_state.chat_messages = []
        st.session_state.voice_prefill = ""
        st.session_state.input_key_counter += 1
        st.session_state.voice_toast_msg = None
        st.session_state.active_tab_right = "matches"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# RIGHT COLUMN: RECOMMENDATIONS, WHAT-IF & ADVISOR VIEWS
# -----------------------------------------------------------------------------
with col_right:
    # Crucial Rule: Hide results completely until user provides details
    if not has_profile_data:
        st.markdown("""
        <div class="qs-empty-state" style="padding: 3.5rem 1.8rem; margin-top: 0.5rem;">
            <div style="font-size: 2.4rem; margin-bottom: 0.6rem; color: #52769B;">📄</div>
            <div class="qs-empty-title" style="font-size: 1.25rem;">Awaiting Your Academic Profile</div>
            <div class="qs-empty-sub" style="font-size: 0.95rem; line-height: 1.5; max-width: 420px; margin: 0.6rem auto 0 auto;">
                Your scholarship matches will appear here once you share your academic profile with the AI.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Action Toolbar inside the right pane
        col_tb1, col_tb2, col_tb3 = st.columns(3)
        with col_tb1:
            btn_matches = st.button(
                "📋 Recommended Matches",
                type="primary" if st.session_state.active_tab_right == "matches" else "secondary",
                use_container_width=True
            )
            if btn_matches:
                st.session_state.active_tab_right = "matches"
                st.rerun()

        with col_tb2:
            btn_whatif = st.button(
                "🔮 What-If Analysis",
                type="primary" if st.session_state.active_tab_right == "what_if" else "secondary",
                use_container_width=True
            )
            if btn_whatif:
                st.session_state.active_tab_right = "what_if"
                st.rerun()

        with col_tb3:
            btn_advisor = st.button(
                "📈 Improvement Advisor",
                type="primary" if st.session_state.active_tab_right == "advisor" else "secondary",
                use_container_width=True
            )
            if btn_advisor:
                st.session_state.active_tab_right = "advisor"
                st.rerun()

        st.markdown("<div style='height: 1px; background: #E2E7ED; margin: 0.75rem 0 1.25rem 0;'></div>", unsafe_allow_html=True)

        # Compute Base Recommendations
        current_recs = ScholarshipScoringEngine.rank_scholarships(
            df=df_dataset,
            profile=prof,
            top_n=5,
            min_score=5.0
        )

        # ---------------------------------------------------------------------
        # SUB-VIEW A: RECOMMENDED MATCHES (DEFAULT RESULTS VIEW)
        # ---------------------------------------------------------------------
        if st.session_state.active_tab_right == "matches":
            col_r_hdr1, col_r_hdr2 = st.columns([3, 1.8])
            with col_r_hdr1:
                st.markdown(f"<div style='font-size: 0.95rem; color: #23364D; padding-top: 4px;'><b>Indexed Matches:</b> <span style='color: #355E85; font-weight: 700;'>{len(current_recs)} Verified Schemes</span></div>", unsafe_allow_html=True)
            with col_r_hdr2:
                if current_recs:
                    csv_export_df = pd.DataFrame([{
                        "Rank": i + 1,
                        "Scholarship_Name": r["name"],
                        "Provider": r["provider"],
                        "Score": r["score"],
                        "Amount": r["amount"],
                        "Portal": r["portal"],
                        "Reasons": "; ".join(r["reasons"])
                    } for i, r in enumerate(current_recs)])
                    st.download_button(
                        label="📥 Export Matches (CSV)",
                        data=csv_export_df.to_csv(index=False).encode('utf-8'),
                        file_name="scholarship_matches.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            if not current_recs:
                st.markdown("""
                <div class="qs-empty-state">
                    <div style="font-size: 2rem; margin-bottom: 0.4rem; color: #52769B;">📄</div>
                    <div class="qs-empty-title">No Direct Matches Found</div>
                    <div class="qs-empty-sub">Try expanding your profile or exploring the <b>What-If Analysis</b> tool to uncover unlocked opportunities.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for idx, scheme in enumerate(current_recs, 1):
                    with st.container():
                        is_high_match = scheme['score'] >= 50.0
                        badge_html = (
                            "<span class='qs-badge-eligible'>✓ Eligible</span>"
                            if is_high_match
                            else "<span class='qs-badge-potential'>⚠ Potential Match</span>"
                        )
                        amount_display = scheme['amount'] if scheme['amount'] and str(scheme['amount']) != 'nan' else 'Financial Assistance / Full Fee Waiver'
                        reasons_html = ''.join([f"<span class='qs-reason-pill'>{r}</span>" for r in scheme['reasons']])

                        st.markdown(f"""
                        <div class="qs-scholarship-card">
                            <div class="qs-card-top">
                                <div>
                                    <span class="qs-provider-tag">🏛️ {scheme['provider']}</span>
                                    <div class="qs-scheme-name">{idx}. {scheme['name']}</div>
                                </div>
                                <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 0.35rem;">
                                    {badge_html}
                                    <span class="qs-score-pill">★ {scheme['score']} pts</span>
                                </div>
                            </div>
                            <div class="qs-benefit-panel">
                                <span>💵</span>
                                <span><b>Verified Financial Assistance:</b> {amount_display}</span>
                            </div>
                            <div>
                                <span style="font-size: 0.78rem; font-weight: 600; color: #687586; display: block; margin-bottom: 0.35rem;">Why You Qualify:</span>
                                {reasons_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col_det, col_act = st.columns([3.8, 1.8])
                        with col_det:
                            with st.expander(f"View Requirements for '{scheme['name']}'"):
                                st.markdown(f"**State / Domicile:** {scheme['state']}")
                                st.markdown(f"**Education Level:** {scheme['education_level']}")
                                st.markdown(f"**Community Eligibility:** {scheme['community']}")
                                if scheme['description'] and str(scheme['description']) != 'nan':
                                    st.markdown(f"**Description:** {scheme['description']}")
                                if scheme['documents'] and str(scheme['documents']) != 'nan':
                                    st.markdown(f"**Documents Required:** {scheme['documents']}")

                        with col_act:
                            target_url = scheme['portal'] if scheme['portal'] and scheme['portal'] != '#' else 'https://scholarships.gov.in'
                            st.link_button(
                                label="Apply on Portal ↗",
                                url=target_url,
                                type="primary",
                                use_container_width=True
                            )

        # ---------------------------------------------------------------------
        # SUB-VIEW B: DEDICATED WHAT-IF ANALYSIS SECTION & DATA VISUALIZATION
        # ---------------------------------------------------------------------
        elif st.session_state.active_tab_right == "what_if":
            st.markdown("""
            <div class="qs-card" style="padding: 1.25rem 1.45rem;">
                <div class="qs-card-title">🔮 What-If Scenario Simulator</div>
                <div class="qs-card-desc">Simulate how adjusting your academic marks, family income, or quota qualifications unlocks higher scores and newly matched schemes.</div>
            </div>
            """, unsafe_allow_html=True)

            col_wi1, col_wi2 = st.columns(2)
            with col_wi1:
                wi_marks = st.slider(
                    "Simulated Academic Score (% or CGPA)",
                    min_value=40.0,
                    max_value=100.0,
                    value=float(prof.get('min_marks_or_cgpa')) if prof.get('min_marks_or_cgpa') else 75.0,
                    step=1.0
                )
                wi_income = st.slider(
                    "Simulated Annual Income (₹)",
                    min_value=50000.0,
                    max_value=800000.0,
                    value=float(prof.get('annual_income')) if prof.get('annual_income') else 200000.0,
                    step=25000.0,
                    format="₹%.0f"
                )
                wi_community = st.selectbox(
                    "Simulated Community Category",
                    options=["BC", "MBC", "DNC", "SC", "ST", "OC", "General", "Minority"],
                    index=["BC", "MBC", "DNC", "SC", "ST", "OC", "General", "Minority"].index(prof.get("community")) if prof.get("community") in ["BC", "MBC", "DNC", "SC", "ST", "OC", "General", "Minority"] else 0
                )

            with col_wi2:
                wi_govt_school = st.selectbox(
                    "Studied in Govt School (7.5% Quota)?",
                    options=["No", "Yes"],
                    index=1 if prof.get("government_school") == "Yes" else 0
                )
                wi_first_grad = st.selectbox(
                    "First Graduate in Family?",
                    options=["No", "Yes"],
                    index=1 if prof.get("first_graduate") == "Yes" else 0
                )
                wi_hostel = st.selectbox(
                    "Accommodation Type",
                    options=["Day Scholar", "Hosteller"],
                    index=1 if prof.get("hosteller") == "Hosteller" else 0
                )

            # Compute What-If Profile
            simulated_profile = prof.copy()
            simulated_profile.update({
                'min_marks_or_cgpa': wi_marks,
                'annual_income': wi_income,
                'community': wi_community,
                'government_school': wi_govt_school,
                'first_graduate': wi_first_grad,
                'hosteller': wi_hostel
            })

            wi_recs = ScholarshipScoringEngine.rank_scholarships(df_dataset, simulated_profile, top_n=5, min_score=5.0)

            current_top_score = current_recs[0]['score'] if current_recs else 0.0
            wi_top_score = wi_recs[0]['score'] if wi_recs else 0.0
            score_delta = wi_top_score - current_top_score

            col_comp1, col_comp2, col_comp3 = st.columns(3)
            with col_comp1:
                st.metric("Base Top Score", f"{current_top_score:.1f} pts")
            with col_comp2:
                st.metric("What-If Top Score", f"{wi_top_score:.1f} pts", delta=f"{score_delta:+.1f} pts")
            with col_comp3:
                st.metric("Total Indexed Matches", f"{len(wi_recs)} Top Schemes")

            # Data Visualization: What-If Scenario Score Comparison Chart
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #23364D; margin: 1.25rem 0 0.5rem 0;'>📊 Scenario Score Comparison (Base vs. Simulated)</div>", unsafe_allow_html=True)
            chart_rows = []
            for wi_item in wi_recs[:5]:
                scheme_name = wi_item['name']
                short_name = scheme_name if len(scheme_name) <= 24 else scheme_name[:22] + "..."
                base_match = next((r['score'] for r in current_recs if r['name'] == scheme_name), 0.0)
                chart_rows.append({"Scholarship": short_name, "Scenario": "Base Score", "Score (pts)": base_match})
                chart_rows.append({"Scholarship": short_name, "Scenario": "What-If Simulated", "Score (pts)": wi_item['score']})

            if chart_rows:
                chart_df = pd.DataFrame(chart_rows)
                wi_chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X('Scholarship:N', title='Scholarship Scheme', sort=None, axis=alt.Axis(labelAngle=-15, labelColor='#23364D', titleColor='#23364D', labelFont='Inter', titleFont='Inter')),
                    y=alt.Y('Score (pts):Q', title='Match Score (pts)', axis=alt.Axis(labelColor='#687586', titleColor='#23364D', gridColor='#E2E7ED', labelFont='Inter', titleFont='Inter')),
                    color=alt.Color('Scenario:N', scale=alt.Scale(
                        domain=['Base Score', 'What-If Simulated'],
                        range=['#52769B', '#355E85']
                    ), legend=alt.Legend(orient='top', title=None, labelFont='Inter', labelColor='#23364D')),
                    xOffset='Scenario:N'
                ).properties(
                    height=260
                ).configure_view(
                    strokeWidth=0
                )
                st.altair_chart(wi_chart, use_container_width=True)

            st.markdown("<div style='font-size: 0.88rem; font-weight: 700; color: #23364D; margin: 1rem 0 0.4rem 0;'>Simulated Schemes & Rank Shifts:</div>", unsafe_allow_html=True)
            for rank_i, wi_item in enumerate(wi_recs, 1):
                is_new = not any(wi_item['name'] == r['name'] for r in current_recs)
                new_tag = " <span class='qs-tip-badge' style='background: #E8EFF6; color: #355E85;'>🌟 Newly Unlocked</span>" if is_new else ""
                st.markdown(f"- **#{rank_i} {wi_item['name']}** — `★ {wi_item['score']} pts` ({wi_item['provider']}){new_tag}", unsafe_allow_html=True)

            if st.button("← Back to Recommended Scholarships", type="secondary"):
                st.session_state.active_tab_right = "matches"
                st.rerun()

        # ---------------------------------------------------------------------
        # SUB-VIEW C: DEDICATED PROFILE IMPROVEMENT ADVISOR & DATA VISUALIZATION
        # ---------------------------------------------------------------------
        elif st.session_state.active_tab_right == "advisor":
            st.markdown("""
            <div class="qs-card" style="padding: 1.25rem 1.45rem;">
                <div class="qs-card-title">📈 Profile Improvement Advisor</div>
                <div class="qs-card-desc">Actionable recommendations to qualify for higher-value scholarships and bridge criteria gaps.</div>
            </div>
            """, unsafe_allow_html=True)

            # Data Visualization: Criteria Gap Analysis vs. High-Value Benchmark Thresholds
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #23364D; margin-bottom: 0.5rem;'>📊 Criteria Gap Analysis (Current Profile vs. High-Value Benchmarks)</div>", unsafe_allow_html=True)

            # Calculate metric benchmarks
            marks_val = float(prof.get('min_marks_or_cgpa') or 0.0)
            marks_normalized = min(100.0, marks_val if marks_val > 10 else marks_val * 10.0)
            
            income_val = prof.get('annual_income')
            if income_val is None:
                income_index = 0.0
            elif income_val <= 250000:
                income_index = 100.0
            elif income_val <= 400000:
                income_index = 75.0
            elif income_val <= 800000:
                income_index = 50.0
            else:
                income_index = 25.0

            quota_score = (50.0 if prof.get('government_school') == 'Yes' else 0.0) + (50.0 if prof.get('first_graduate') == 'Yes' else 0.0)

            advisor_chart_rows = [
                {"Criterion": "Academic Merit", "Metric": "Current Profile", "Level (%)": marks_normalized},
                {"Criterion": "Academic Merit", "Metric": "High-Value Target", "Level (%)": 60.0},
                {"Criterion": "Income Ceiling", "Metric": "Current Profile", "Level (%)": income_index},
                {"Criterion": "Income Ceiling", "Metric": "High-Value Target", "Level (%)": 80.0},
                {"Criterion": "Quota Optimization", "Metric": "Current Profile", "Level (%)": quota_score},
                {"Criterion": "Quota Optimization", "Metric": "High-Value Target", "Level (%)": 50.0},
                {"Criterion": "Profile Calibration", "Metric": "Current Profile", "Level (%)": float(completion_score)},
                {"Criterion": "Profile Calibration", "Metric": "High-Value Target", "Level (%)": 100.0},
            ]
            adv_chart_df = pd.DataFrame(advisor_chart_rows)
            adv_chart = alt.Chart(adv_chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X('Criterion:N', title=None, axis=alt.Axis(labelAngle=0, labelColor='#23364D', labelFont='Inter')),
                y=alt.Y('Level (%):Q', title='Compliance / Target Level (%)', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelColor='#687586', titleColor='#23364D', gridColor='#E2E7ED', labelFont='Inter')),
                color=alt.Color('Metric:N', scale=alt.Scale(
                    domain=['Current Profile', 'High-Value Target'],
                    range=['#52769B', '#355E85']
                ), legend=alt.Legend(orient='top', title=None, labelFont='Inter', labelColor='#23364D')),
                xOffset='Metric:N'
            ).properties(
                height=250
            ).configure_view(
                strokeWidth=0
            )
            st.altair_chart(adv_chart, use_container_width=True)

            # Actionable Guidance Cards
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #23364D; margin: 1.25rem 0 0.5rem 0;'>💡 Actionable Improvement Steps</div>", unsafe_allow_html=True)
            actionable_tips = ScholarshipImprovementAdvisor.generate_guidance(prof, current_recs)

            if not actionable_tips:
                st.markdown("""
                <div style="background: #E2F3E8; border: 1px solid #C3E6D0; color: #246B45; padding: 0.85rem 1.15rem; border-radius: 10px; font-size: 0.88rem; font-weight: 500;">
                    ✓ Your profile details are complete with criteria aligned to top available schemes.
                </div>
                """, unsafe_allow_html=True)
            else:
                for tip in actionable_tips:
                    st.markdown(f"""
                    <div class="qs-tip-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                            <div class="qs-tip-title">
                                {tip['icon']} {tip['title']}
                            </div>
                            <span class="qs-tip-badge">
                                {tip['impact']}
                            </span>
                        </div>
                        <div class="qs-tip-desc">
                            {tip['description']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            if st.button("← Back to Recommended Scholarships", type="secondary"):
                st.session_state.active_tab_right = "matches"
                st.rerun()

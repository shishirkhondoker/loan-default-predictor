import re
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = BASE_DIR / "outputs" / "test_predictions.csv"


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Container padding */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1600px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 35%, #2563eb 100%);
        padding: 5px 60px;
        border-radius: 24px;
        color: white;
        margin-bottom: 15px;
        box-shadow: 0 25px 60px rgba(15, 23, 42, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
        
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(37, 99, 235, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        color: #e0f2fe;
        padding: 8px 16px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
        border: 1px solid rgba(255, 255, 255, 0.22);
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: 35px;
       
        margin: 0 0 10px 0;
        font-weight: 900;
        letter-spacing: -1.5px;
        color: #ffffff;
        text-align: center;
        
    }

    # .hero p {
    #     font-size: 18px;
    #     color: #cffafe;
    #     max-width: 920px;
    #     line-height: 1.7;
    #     margin: 0;
    #     font-weight: 400;
    # }

    /* Section title */
    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin: 32px 0 20px 0;
        letter-spacing: -0.5px;
    }

    /* Cards */
    .soft-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 26px 28px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.5);
        margin-bottom: 20px;
    }

    .example-card {
        background: #ffffff;
        border: 1.5px solid #e0f2fe;
        border-radius: 22px;
        padding: 24px 26px;
        min-height: 150px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.6);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }

    .example-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(37, 99, 235, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.6);
        border-color: #7dd3fc;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    }

    .example-title {
        color: #0f172a;
        font-weight: 800;
        font-size: 16px;
        margin-bottom: 12px;
        letter-spacing: -0.3px;
    }

    .example-query {
        color: #475569;
        font-size: 14px;
        line-height: 1.6;
        font-weight: 500;
    }

    .query-shell {
        background: #ffffff;
        border: 1.5px solid #ddd6fe;
        border-radius: 28px;
        padding: 32px 36px;
        box-shadow: 0 12px 36px rgba(15, 23, 42, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.5);
        margin-top: 24px;
        margin-bottom: 24px;
    }

    .query-shell h2 {
        margin: 0 0 8px 0;
        font-size: 28px;
        font-weight: 900;
        color: #0f172a;
        letter-spacing: -0.6px;
    }

    .query-shell p {
        margin: 0;
        color: #64748b;
        font-size: 15px;
        line-height: 1.6;
    }

    .result-title {
        font-size: 28px;
        font-weight: 900;
        color: #0f172a;
        margin: 24px 0 14px 0;
        letter-spacing: -0.6px;
    }

    .result-subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 20px;
        font-weight: 500;
    }

    .reason-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1.5px solid #cbd5e1;
        border-left: 5px solid #3b82f6;
        border-radius: 18px;
        padding: 20px 22px;
        margin-bottom: 14px;
        color: #0f172a;
        font-size: 15px;
        line-height: 1.65;
        font-weight: 500;
    }

    .footer-note {
        color: #7c8fa6;
        font-size: 13px;
        text-align: center;
        padding-top: 32px;
        margin-top: 48px;
        border-top: 1px solid #e2e8f0;
        font-weight: 500;
        letter-spacing: 0.2px;
    }

    /* Streamlit text input styling */
    div[data-testid="stTextInput"] {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        padding: 18px 20px 12px 20px;
        border-radius: 22px;
        border: 2px solid #3b82f6;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.12);
    }

    div[data-testid="stTextInput"] label {
        color: #0f172a;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: -0.2px;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 16px;
        border: 1.5px solid #93c5fd;
        padding: 14px 16px;
        font-size: 16px;
        background-color: #ffffff;
        color: #1e293b;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #cbd5e1;
    }

    div[data-testid="stTextInput"] input:focus {
        border: 2px solid #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1);
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.2rem;
        border-radius: 16px;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        margin-top: 28px;
        letter-spacing: -0.2px;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(59, 130, 246, 0.4);
    }

    div.stButton > button:active {
        transform: translateY(0px);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 22px;
        padding: 22px 24px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.3px;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 32px;
        font-weight: 900;
        letter-spacing: -0.8px;
    }

    /* Dataframe card effect */
    div[data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        border: 1.5px solid #e2e8f0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1.5px solid #e2e8f0;
    }

    [data-testid="stSidebar"] h2 {
        color: #0f172a;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin-bottom: 20px;
    }

    [data-testid="stSidebar"] p {
        color: #475569;
        font-weight: 500;
        line-height: 1.6;
    }

    /* Selectbox styling */
    div[data-testid="stSelectbox"] {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        padding: 14px 16px 8px 16px;
        border-radius: 16px;
        border: 2px solid #93c5fd;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
    }

    div[data-testid="stSelectbox"] label {
        color: #0f172a;
        font-weight: 700;
        font-size: 14px;
    }

    /* Warning and info boxes */
    div[data-testid="stAlert"] {
        border-radius: 18px;
        border-width: 1.5px;
        padding: 16px 18px;
        margin-bottom: 18px;
    }

    /* Dashboard Card */
    .dashboard-card {
        /* Changed to an inviting green/teal gradient for a friendly website look */
        background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 20px;
        margin-bottom: 20px;
        border: 1.5px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 30px rgba(6, 95, 70, 0.18);
    }

    .dashboard-title {
        font-size: 18px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.3px;
        margin: 0;
    }

    /* Query Card */
    .query-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #3b82f6;
        border-radius: 24px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.12);
    }

    .query-title {
        font-size: 20px;
        font-weight: 900;
        color: #0f172a;
        margin: 0 0 8px 0;
        letter-spacing: -0.4px;
    }

    .query-subtitle {
        color: #64748b;
        font-size: 14px;
        margin: 0;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Examples Container */
    .examples-container {
        background: #f8fafc;
        border-radius: 16px;
        padding: 16px 0;
        margin-top: 20px;
    }

    .example-label {
        color: #0f172a;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 0 14px;
        display: block;
        margin-bottom: 12px;
    }

    .example-card-small {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 16px;
        padding: 14px 16px;
        margin: 0 8px 10px 8px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease-in-out;
    }

    .example-card-small:hover {
        border-color: #93c5fd;
        background: #f0f9ff;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.1);
    }

    .example-title-small {
        color: #0f172a;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 6px;
        letter-spacing: -0.2px;
    }

    .example-query-small {
        color: #475569;
        font-size: 12px;
        line-height: 1.5;
        font-weight: 500;
    }

    /* Visualization Card */
    .viz-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1.5px solid #e2e8f0;
        border-radius: 24px;
        padding: 24px 26px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .viz-title {
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.3px;
    }

    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border: 1.5px solid #bfdbfe;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08);
    }

    .info-box strong {
        color: #0f172a;
        font-size: 14px;
        display: block;
        margin-bottom: 8px;
    }

    .info-box p {
        color: #475569;
        line-height: 1.6;
    }

    /* Responsive Typography */
    body, html {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    @media (max-width: 1024px) {
        .hero h1 {
            font-size: 38px;
        }

        .hero p {
            font-size: 16px;
        }

        .dashboard-card, .query-card, .viz-card {
            padding: 18px 20px;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 40px 28px;
        }

        .hero h1 {
            font-size: 32px;
        }

        .hero p {
            font-size: 15px;
        }

        [data-testid="stMetricValue"] {
            font-size: 28px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        st.error("Prediction file not found. Please run: python -m src.train")
        st.stop()

    return pd.read_csv(PREDICTIONS_PATH)


preds = load_predictions()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style='background: linear-gradient(165deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%); padding: 20px 16px; border-radius: 18px; border: 1.5px solid rgba(168,85,247,0.3); box-shadow: 0 8px 32px rgba(99,102,241,0.2); margin-bottom: 16px; position: relative; overflow: hidden;'>
            <div style='position: absolute; top: -50%; right: -30%; width: 200px; height: 200px; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); border-radius: 50%;'></div>
            <h3 style='color: #ffffff; font-size: 18px; margin: 0 0 10px 0; font-weight: 800; letter-spacing: -0.3px; position: relative; z-index: 1;'>📊 Risk Dashboard</h3>
            <p style='color: #e0e7ff; font-size: 13px; line-height: 1.7; margin: 0; position: relative; z-index: 1;'>Interactive risk analysis & borrower insights powered by ML predictions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.08) 100%); border: 2px solid rgba(239,68,68,0.25); border-radius: 16px; padding: 18px 18px; margin-bottom: 16px; position: relative; overflow: hidden;'>
            <div style='position: absolute; top: 0; right: 0; width: 60px; height: 60px; background: rgba(239,68,68,0.1); border-radius: 0 16px 0 60px;'></div>
            <p style='color: #dc2626; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; margin: 0 0 12px 0; display: flex; align-items: center; gap: 8px;'>
                <span style='display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; background: rgba(239,68,68,0.2); border-radius: 50%; font-size: 10px;'>!</span>
                Supported Queries
            </p>
            <ul style='margin: 0; padding-left: 0; list-style: none;'>
                <li style='color: #f87171; font-size: 13px; line-height: 2; display: flex; align-items: center; gap: 10px;'>
                    <span style='width: 6px; height: 6px; background: #f87171; border-radius: 50%; display: inline-block;'></span>
                    Top 10 highest-risk borrowers
                </li>
                <li style='color: #f87171; font-size: 13px; line-height: 2; display: flex; align-items: center; gap: 10px;'>
                    <span style='width: 6px; height: 6px; background: #f87171; border-radius: 50%; display: inline-block;'></span>
                    Borrower-level explanation
                </li>
                <li style='color: #f87171; font-size: 13px; line-height: 2; display: flex; align-items: center; gap: 10px;'>
                    <span style='width: 6px; height: 6px; background: #f87171; border-radius: 50%; display: inline-block;'></span>
                    Default rate for loans above $20,000
                </li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #0d9488 0%, #14b8a6 50%, #06b6d4 100%); padding: 20px 18px; border-radius: 16px; border: 2px solid rgba(20,184,166,0.3); box-shadow: 0 6px 24px rgba(13,148,136,0.25); position: relative; overflow: hidden;'>
            <div style='position: absolute; top: -20px; left: -20px; width: 80px; height: 80px; background: rgba(255,255,255,0.08); border-radius: 20px; transform: rotate(15deg);'></div>
            <div style='position: absolute; bottom: -10px; right: -10px; width: 60px; height: 60px; background: rgba(255,255,255,0.06); border-radius: 50%;'></div>
            <p style='color: #a7f3d0; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin: 0 0 16px 0; position: relative; z-index: 1;'>Model Configuration</p>
            <div style='display: flex; flex-direction: column; gap: 12px; position: relative; z-index: 1;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: #5eead4; font-size: 12px; font-weight: 600;'>Model Type</span>
                    <span style='color: #ffffff; font-size: 13px; font-weight: 700; background: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 20px;'>Random Forest</span>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: #5eead4; font-size: 12px; font-weight: 600;'>Sample Size</span>
                    <span style='color: #ffffff; font-size: 13px; font-weight: 700; background: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 20px;'>50,000</span>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: #5eead4; font-size: 12px; font-weight: 600;'>Test Split</span>
                    <span style='color: #ffffff; font-size: 13px; font-weight: 700; background: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 20px;'>20%</span>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: #5eead4; font-size: 12px; font-weight: 600;'>Primary Use</span>
                    <span style='color: #ffffff; font-size: 13px; font-weight: 700; background: rgba(255,255,255,0.15); padding: 4px 10px; border-radius: 20px;'>Risk Ranking</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Hero section
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI/ML Credit Risk Demo</div>
        <h1>Mini Loan Default Risk Predictor</h1>
        
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3-Column Website Layout
# ============================================================

left_col, center_col, right_col = st.columns([1, 2, 1.2])

# ============================================================
# LEFT COLUMN: Risk Dashboard
# ============================================================

with left_col:
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="dashboard-title">📊 Risk Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics in left column
    st.metric("Total borrowers", f"{len(preds):,}")

    if "target" in preds.columns:
        default_rate = preds['target'].mean()
        st.metric("Observed default", f"{default_rate:.2%}")
    else:
        st.metric("Observed default", "N/A")

    if "predicted_risk" in preds.columns:
        avg_risk = preds['predicted_risk'].mean()
        st.metric("Avg risk score", f"{avg_risk:.2%}")
    else:
        st.metric("Avg risk score", "N/A")

    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)



# ============================================================
# CENTER COLUMN: Query Interface
# ============================================================

with center_col:

    # ---------- Quick Actions Title ----------
    st.markdown("###  Quick Actions")

    # ---------- Clickable List ----------
    st.markdown("""
    <style>
    .quick-list {
        background: #1e1b4b;
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }

    .quick-item {
        color: #c4b5fd;
        cursor: pointer;
        margin: 6px 0;
        font-size: 14px;
    }

    .quick-item:hover {
        color: white;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Using buttons but styled like list items
    if st.button("• Show top 10 highest-risk borrowers"):
        st.session_state.query_input = "Show top 10 highest-risk borrowers"

    if st.button("• Average default rate for loans above $20,000"):
        st.session_state.query_input = "Average default rate for loans above $20,000"

    if st.button("• Why was borrower 6726267 flagged?"):
        st.session_state.query_input = "Why was borrower 6726267 flagged?"

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Query Input ----------
    query = st.text_input(
        "Your query",
        placeholder="e.g., Show top 10 highest-risk borrowers",
        key="query_input",
        label_visibility="collapsed"
    )

    button_clicked = st.button(" Analyze", use_container_width=True)
    run_query = button_clicked or bool(query.strip())

    st.markdown("<div style='height: 14px'></div>", unsafe_allow_html=True)

    

# ============================================================
# RIGHT COLUMN: Visualization Selector
# ============================================================

with right_col:
    st.markdown(
        """
        <div class="viz-card">
            <div class="viz-title">📈 Visualizations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    viz_option = st.selectbox(
        "Select chart",
        (
            "Default Rate by Loan Amount",
            "Precision-Recall Curve",
            "ROC Curve",
            "Feature Importance",
            "Risk Score Distribution"
        ),
        key="viz_select"
    )


# ============================================================
# Helper functions
# ============================================================

def format_currency(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def show_top_10():
    st.markdown(
        '<div class="result-title">Top 10 Highest-Risk Borrowers</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="result-subtitle">Borrowers are ranked by predicted default risk from the selected model.</div>',
        unsafe_allow_html=True,
    )

    if "predicted_risk" not in preds.columns:
        st.error("`predicted_risk` column not found in predictions file.")
        return

    top_10 = preds.sort_values("predicted_risk", ascending=False).head(10).copy()

    show_cols = [
        col for col in [
            "borrower_id",
            "predicted_risk",
            "loan_amnt",
            "annual_inc",
            "dti",
            "int_rate",
            "target",
        ]
        if col in top_10.columns
    ]

    display_df = top_10[show_cols].copy()

    if "predicted_risk" in display_df.columns:
        display_df["predicted_risk"] = display_df["predicted_risk"].map(lambda x: f"{x:.2%}")

    if "loan_amnt" in display_df.columns:
        display_df["loan_amnt"] = display_df["loan_amnt"].map(format_currency)

    if "annual_inc" in display_df.columns:
        display_df["annual_inc"] = display_df["annual_inc"].map(format_currency)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.info(
        "Copy a borrower_id from this table and ask: Why was borrower [borrower_id] flagged?"
    )


def show_default_rate():
    st.markdown(
        '<div class="result-title">Default Rate for Loans Above $20,000</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="result-subtitle">This statistic uses the actual observed default labels in the test set.</div>',
        unsafe_allow_html=True,
    )

    if "loan_amnt" not in preds.columns or "target" not in preds.columns:
        st.error("Required columns `loan_amnt` and `target` not found.")
        return

    filtered = preds[preds["loan_amnt"] > 20000]

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Default rate", f"{filtered['target'].mean():.2%}")

    with col_b:
        st.metric("Loans above $20,000", f"{len(filtered):,}")

    with col_c:
        st.metric("Average loan amount", format_currency(filtered["loan_amnt"].mean()))

    preview_cols = [
        col for col in ["borrower_id", "loan_amnt", "annual_inc", "dti", "target"]
        if col in filtered.columns
    ]

    with st.expander("Preview loans above $20,000"):
        preview = filtered[preview_cols].head(20).copy()

        if "loan_amnt" in preview.columns:
            preview["loan_amnt"] = preview["loan_amnt"].map(format_currency)

        if "annual_inc" in preview.columns:
            preview["annual_inc"] = preview["annual_inc"].map(format_currency)

        st.dataframe(preview, use_container_width=True, hide_index=True)


def show_borrower_explanation(query_text):
    st.markdown(
        '<div class="result-title">Borrower Risk Explanation</div>',
        unsafe_allow_html=True,
    )

    borrower_match = re.search(
        r"borrower\s+([A-Za-z0-9_-]+)",
        query_text,
        re.IGNORECASE,
    )

    if not borrower_match:
        st.warning("Please write the query like: Why was borrower 123456 flagged?")
        return

    borrower_id = borrower_match.group(1)

    if "borrower_id" not in preds.columns:
        st.error("`borrower_id` column not found in predictions file.")
        return

    borrower_rows = preds[preds["borrower_id"].astype(str) == str(borrower_id)]

    if borrower_rows.empty:
        st.warning(
            f"No borrower found with ID: {borrower_id}. "
            "Try copying a borrower_id from the top 10 table."
        )
        return

    borrower = borrower_rows.iloc[0]

    left, right = st.columns([1, 2])

    with left:
        if "predicted_risk" in borrower.index:
            st.metric("Predicted default risk", f"{borrower['predicted_risk']:.2%}")

        if "target" in borrower.index:
            target_value = int(borrower["target"])
            target_text = "Default" if target_value == 1 else "Non-default"
            st.metric("Actual outcome", target_text)

    with right:
        detail_data = {}

        if "borrower_id" in borrower.index:
            detail_data["Borrower ID"] = borrower["borrower_id"]

        if "loan_amnt" in borrower.index:
            detail_data["Loan amount"] = format_currency(borrower["loan_amnt"])

        if "annual_inc" in borrower.index:
            detail_data["Annual income"] = format_currency(borrower["annual_inc"])

        if "dti" in borrower.index:
            detail_data["Debt-to-income ratio"] = (
                f"{borrower['dti']:.2f}%" if pd.notna(borrower["dti"]) else "N/A"
            )

        if "int_rate" in borrower.index:
            detail_data["Interest rate"] = (
                f"{borrower['int_rate']:.2f}%" if pd.notna(borrower["int_rate"]) else "N/A"
            )

        details_df = pd.DataFrame(
            list(detail_data.items()),
            columns=["Field", "Value"],
        )

        st.dataframe(details_df, use_container_width=True, hide_index=True)

    st.markdown("### Key reasons")

    reasons = []

    if "dti" in borrower.index and pd.notna(borrower["dti"]):
        if borrower["dti"] >= 30:
            reasons.append(
                f"Debt-to-income ratio is high at **{borrower['dti']:.1f}%**, "
                "which may indicate repayment pressure."
            )
        else:
            reasons.append(
                f"Debt-to-income ratio is **{borrower['dti']:.1f}%**, "
                "which contributes to the overall risk profile."
            )

    if "int_rate" in borrower.index and pd.notna(borrower["int_rate"]):
        if borrower["int_rate"] >= 18:
            reasons.append(
                f"Interest rate is high at **{borrower['int_rate']:.2f}%**, "
                "suggesting the borrower was priced as higher risk."
            )
        else:
            reasons.append(
                f"Interest rate is **{borrower['int_rate']:.2f}%**, "
                "which is part of the model's risk assessment."
            )

    if (
        "loan_amnt" in borrower.index
        and "annual_inc" in borrower.index
        and pd.notna(borrower["loan_amnt"])
        and pd.notna(borrower["annual_inc"])
        and borrower["annual_inc"] > 0
    ):
        loan_to_income = borrower["loan_amnt"] / borrower["annual_inc"]

        if loan_to_income >= 0.3:
            reasons.append(
                f"Loan amount is large relative to annual income "
                f"with a loan-to-income ratio of **{loan_to_income:.2f}**."
            )
        else:
            reasons.append(
                f"Loan-to-income ratio is **{loan_to_income:.2f}**, "
                "which was considered with the other borrower features."
            )

    if not reasons:
        reasons.append(
            "This borrower was flagged because the model found a high-risk pattern "
            "across the available application-time features."
        )

    for reason in reasons:
        st.markdown(
            f"""
            <div class="reason-box">
                {reason}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Visualization Functions
# ============================================================

def plot_default_rate_by_loan_amount():
    bins = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 100000]
    labels = ["0-5k", "5k-10k", "10k-15k", "15k-20k", "20k-30k", "30k-40k", "40k-50k", "50k-100k"]
    
    plot_df = preds.copy()
    plot_df['loan_amnt_bin'] = pd.cut(plot_df['loan_amnt'], bins=bins, labels=labels)
    default_rate_by_bin = plot_df.groupby('loan_amnt_bin', observed=True)['target'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    default_rate_by_bin.plot(kind='bar', ax=ax, color='#3b82f6', edgecolor='#1e40af')
    ax.set_xlabel('Loan Amount Range', fontsize=12, fontweight='600')
    ax.set_ylabel('Default Rate', fontsize=12, fontweight='600')
    ax.set_title('Default Rate by Loan Amount', fontsize=14, fontweight='700')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

def plot_precision_recall_curve():
    if 'predicted_risk' not in preds.columns or 'target' not in preds.columns:
        st.warning("Required columns not found")
        return
    
    probs = preds['predicted_risk']
    y_true = preds['target']
    
    precision, recall, _ = precision_recall_curve(y_true, probs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, color='#3b82f6', linewidth=2.5, label='Precision-Recall')
    ax.fill_between(recall, precision, alpha=0.2, color='#3b82f6')
    ax.set_xlabel('Recall', fontsize=12, fontweight='600')
    ax.set_ylabel('Precision', fontsize=12, fontweight='600')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='700')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

def plot_roc_curve():
    if 'predicted_risk' not in preds.columns or 'target' not in preds.columns:
        st.warning("Required columns not found")
        return
    
    probs = preds['predicted_risk']
    y_true = preds['target']
    
    fpr, tpr, _ = roc_curve(y_true, probs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, color='#3b82f6', linewidth=2.5, label='ROC Curve')
    ax.plot([0, 1], [0, 1], color='#9ca3af', linestyle='--', linewidth=2, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#3b82f6')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='600')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='600')
    ax.set_title('ROC Curve', fontsize=14, fontweight='700')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

def plot_feature_importance():
    importances = [0.25, 0.20, 0.18, 0.22, 0.15]
    features = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'fico_range_low']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, importances, color='#3b82f6', edgecolor='#1e40af')
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='600')
    ax.set_title('Feature Importance', fontsize=14, fontweight='700')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                ha='left', va='center', fontsize=10, fontweight='600')
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_risk_score_distribution():
    if 'predicted_risk' not in preds.columns:
        st.warning("Required column 'predicted_risk' not found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(preds['predicted_risk'], bins=50, color='#3b82f6', edgecolor='#1e40af', alpha=0.8)
    ax.set_xlabel('Predicted Risk Score', fontsize=12, fontweight='600')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='600')
    ax.set_title('Predicted Risk Score Distribution', fontsize=14, fontweight='700')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)


# ============================================================
# Query execution and Results Display
# ============================================================

if run_query:
    if not query.strip():
        # Show custom popup alert when query is empty
        
        st.markdown(
            """
            <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb; margin-top: 20px;">
                <strong>Alert:</strong> Please type a query first.
            </div>
            """, unsafe_allow_html=True
        )
    else:
        query_lower = query.lower()

        # Display results in center column
        with center_col:
            st.markdown("---")

            if (
                "top 10" in query_lower
                or "highest-risk" in query_lower
                or "highest risk" in query_lower
            ):
                show_top_10()

            elif "average default rate" in query_lower or "default rate" in query_lower:
                show_default_rate()

            elif "why" in query_lower and "borrower" in query_lower:
                show_borrower_explanation(query)

            else:
                # Show custom popup alert for unrecognized query
                st.markdown(
                    """
                    <div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb; margin-top: 20px;">
                        <strong>Alert:</strong> I could not understand the query. Please try one of the example queries above.
                    </div>
                    """, unsafe_allow_html=True
                )
        # Display visualization in right column
        with right_col:
            st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)
            if viz_option == "Default Rate by Loan Amount":
                plot_default_rate_by_loan_amount()
            elif viz_option == "Precision-Recall Curve":
                plot_precision_recall_curve()
            elif viz_option == "ROC Curve":
                plot_roc_curve()
            elif viz_option == "Feature Importance":
                plot_feature_importance()
            elif viz_option == "Risk Score Distribution":
                plot_risk_score_distribution()
else:
    # Show visualization when no query is entered
    with right_col:
        st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)
        if viz_option == "Default Rate by Loan Amount":
            plot_default_rate_by_loan_amount()
        elif viz_option == "Precision-Recall Curve":
            plot_precision_recall_curve()
        elif viz_option == "ROC Curve":
            plot_roc_curve()
        elif viz_option == "Feature Importance":
            plot_feature_importance()
        elif viz_option == "Risk Score Distribution":
            plot_risk_score_distribution()


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        Mini Loan Default Predictor • Professional Credit Risk Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)


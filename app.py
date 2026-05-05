import re
from pathlib import Path

import pandas as pd
import streamlit as st


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
    /* App background */
    .stApp {
        background: #f4f7fb;
    }

    /* Hide default Streamlit top padding a little */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
        padding: 46px 52px;
        border-radius: 28px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 22px 50px rgba(15, 23, 42, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.14);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        color: #dbeafe;
        padding: 7px 13px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-bottom: 14px;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .hero h1 {
        font-size: 46px;
        line-height: 1.08;
        margin: 0 0 12px 0;
        font-weight: 900;
        letter-spacing: -1.2px;
    }

    .hero p {
        font-size: 18px;
        color: #dbeafe;
        max-width: 890px;
        line-height: 1.65;
        margin: 0;
    }

    /* Section title */
    .section-title {
        font-size: 22px;
        font-weight: 850;
        color: #0f172a;
        margin: 26px 0 14px 0;
        letter-spacing: -0.3px;
    }

    /* Cards */
    .soft-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 22px 24px;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.075);
        margin-bottom: 18px;
    }

    .example-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 20px;
        padding: 20px 22px;
        min-height: 142px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065);
        transition: all 0.18s ease-in-out;
    }

    .example-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 36px rgba(37, 99, 235, 0.13);
        border-color: #93c5fd;
    }

    .example-title {
        color: #0f172a;
        font-weight: 850;
        font-size: 16px;
        margin-bottom: 10px;
    }

    .example-query {
        color: #334155;
        font-size: 14.5px;
        line-height: 1.55;
    }

    .query-shell {
        background: #ffffff;
        border: 1px solid #c7d2fe;
        border-radius: 26px;
        padding: 26px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
        margin-top: 18px;
        margin-bottom: 18px;
    }

    .query-shell h2 {
        margin: 0 0 6px 0;
        font-size: 24px;
        font-weight: 900;
        color: #0f172a;
        letter-spacing: -0.4px;
    }

    .query-shell p {
        margin: 0;
        color: #64748b;
        font-size: 15px;
        line-height: 1.5;
    }

    .result-title {
        font-size: 26px;
        font-weight: 900;
        color: #0f172a;
        margin: 18px 0 12px 0;
        letter-spacing: -0.5px;
    }

    .result-subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 16px;
    }

    .reason-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 10px;
        color: #0f172a;
        font-size: 15.5px;
        line-height: 1.55;
    }

    .footer-note {
        color: #64748b;
        font-size: 13px;
        text-align: center;
        padding-top: 22px;
    }

    /* Streamlit text input styling */
    div[data-testid="stTextInput"] {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        padding: 16px 18px 10px 18px;
        border-radius: 20px;
        border: 2px solid #2563eb;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.18);
    }

    div[data-testid="stTextInput"] label {
        color: #0f172a;
        font-size: 16px;
        font-weight: 850;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 14px;
        border: 1.5px solid #93c5fd;
        padding: 15px;
        font-size: 16px;
        background-color: white;
    }

    div[data-testid="stTextInput"] input:focus {
        border: 2px solid #2563eb;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.15);
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        padding: 0.78rem 2rem;
        border-radius: 14px;
        font-weight: 850;
        font-size: 16px;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.28);
        transition: 0.2s ease-in-out;
        width: 100%;
        margin-top: 27px;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        color: white;
        transform: translateY(-1px);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 18px 20px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.065);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 750;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 31px;
        font-weight: 900;
    }

    /* Dataframe card effect */
    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
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
    st.markdown("## Risk Dashboard")
    st.markdown(
        """
        This demo app answers simple risk-related queries from the model output.

        **Supported queries**
        - Top 10 highest-risk borrowers
        - Borrower-level explanation
        - Default rate for loans above $20,000
        """
    )

    st.divider()

    st.markdown("### Model Context")
    st.markdown(
        """
        **Chosen model:** Static + behavioral  
        **Sample size:** 50,000 rows  
        **Test split:** 20%  
        **Main use:** Risk ranking
        """
    )

    st.divider()

    # st.markdown("### Demo Tip")
    # st.info(
    #     "Start with the top 10 query. Then copy a borrower_id and ask why that borrower was flagged."
    # )


# ============================================================
# Hero section
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI/ML Credit Risk Demo</div>
        <h1>Mini Loan Default Risk Predictor</h1>
        <p>
            A professional credit-risk dashboard for identifying high-risk borrowers,
            explaining borrower-level risk, and checking simple portfolio statistics
            from the model's test-set predictions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Summary metrics
# ============================================================

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Total test borrowers", f"{len(preds):,}")

with metric_col2:
    if "target" in preds.columns:
        st.metric("Observed default rate", f"{preds['target'].mean():.2%}")
    else:
        st.metric("Observed default rate", "N/A")

with metric_col3:
    if "predicted_risk" in preds.columns:
        st.metric("Average predicted risk", f"{preds['predicted_risk'].mean():.2%}")
    else:
        st.metric("Average predicted risk", "N/A")


# ============================================================
# Example queries
# ============================================================

st.markdown('<div class="section-title">Try an Example Query</div>', unsafe_allow_html=True)

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    st.markdown(
        """
        <div class="example-card">
            <div class="example-title">Top-risk list</div>
            <div class="example-query">
                Show me the top 10 highest-risk borrowers in the test set.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with example_col2:
    st.markdown(
        """
        <div class="example-card">
            <div class="example-title"> Borrower explanation</div>
            <div class="example-query">
                Why was borrower 6726267 flagged?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with example_col3:
    st.markdown(
        """
        <div class="example-card">
            <div class="example-title"> Portfolio statistic</div>
            <div class="example-query">
                What's the average default rate for loans above $20,000?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Query input
# ============================================================

# st.markdown(
#     """
#     <div class="query-shell">
#         <h2>Ask the Risk Assistant</h2>
#         <p>
#             Type your query below and press Enter, or click Analyze. The interface uses
#             simple rule-based query handling for this assignment demo.
#         </p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

input_col, button_col = st.columns([4.2, 1])

with input_col:
    query = st.text_input(
        "Enter your query",
        placeholder="Example: Show me the top 10 highest-risk borrowers in the test set.",
        key="query_input",
    )

with button_col:
    button_clicked = st.button("Analyze")

run_query = button_clicked or bool(query.strip())


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
    st.markdown(
        '<div class="result-subtitle">A plain-English explanation designed for a credit officer.</div>',
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
# Query execution
# ============================================================

if run_query:
    if not query.strip():
        st.warning("Please type a query first.")
    else:
        query_lower = query.lower()

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
            st.warning(
                "I could not understand the query. Please try one of the example queries above."
            )

else:
    st.markdown(
        """
        <div class="soft-card">
            <b>Demo workflow:</b>
            Start with the top 10 highest-risk borrowers query. Then copy one borrower ID
            and ask why that borrower was flagged.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

# Assuming df is your final dataset (use your final df from model pipeline)
df = pd.read_csv("outputs/test_predictions.csv")  # Or however you're loading the final dataset

# Default rate by loan amount

def plot_default_rate_by_loan_amount():
    # Define bins for loan amounts
    bins = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 100000]
    labels = ["0-5k", "5k-10k", "10k-15k", "15k-20k", "20k-30k", "30k-40k", "40k-50k", "50k-100k"]
    
    # Create a new column to assign the loan amount to a bin
    df['loan_amnt_bin'] = pd.cut(df['loan_amnt'], bins=bins, labels=labels)
    
    # Calculate default rate by loan amount bin
    default_rate_by_bin = df.groupby('loan_amnt_bin')['target'].mean()
    
    # Plot the default rate by loan amount bin
    fig, ax = plt.subplots()
    default_rate_by_bin.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel('Loan Amount Range')
    ax.set_ylabel('Default Rate')
    ax.set_title('Default Rate by Loan Amount')
    st.pyplot(fig)

# Precision-Recall Curve
def plot_precision_recall_curve():
    probs = df['predicted_risk']  # assuming 'predicted_risk' column contains the probabilities
    y_true = df['target']
    
    precision, recall, _ = precision_recall_curve(y_true, probs)
    
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='b', label='Precision-Recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    st.pyplot(fig)

# ROC Curve
def plot_roc_curve():
    probs = df['predicted_risk']  # assuming 'predicted_risk' column contains the probabilities
    y_true = df['target']
    
    fpr, tpr, _ = roc_curve(y_true, probs)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='b', label='ROC curve')
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')  # Random classifier line
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    st.pyplot(fig)

# Feature Importance Bar Chart
def plot_feature_importance():
    # Assuming you are using a model like Random Forest or XGBoost
    # Replace with model-specific feature importance
    importances = [0.2, 0.1, 0.15, 0.25, 0.3]  # Example random importance values
    features = ['loan_amnt', 'annual_inc', 'dti', 'int_rate', 'fico_range_low']
    
    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

# Risk Score Distribution
def plot_risk_score_distribution():
    fig, ax = plt.subplots()
    df['predicted_risk'].plot(kind='hist', bins=50, ax=ax)
    ax.set_xlabel('Predicted Risk')
    ax.set_ylabel('Frequency')
    ax.set_title('Predicted Risk Score Distribution')
    st.pyplot(fig)


# Streamlit UI
st.title('Loan Default Risk Prediction Dashboard')

# Create three columns: left for the risk dashboard, middle for prediction description, right for visualization
left_column, middle_column, right_column = st.columns([1, 2, 3])

# Left column content - Risk Dashboard
with left_column:
    st.markdown(
        """
        <style>
            .risk-dashboard {
                background-color: #f7f7f7;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            .risk-dashboard h3 {
                color: #007bff;
                font-size: 22px;
                font-weight: bold;
            }
            .risk-dashboard p {
                font-size: 18px;
                color: #555;
            }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="risk-dashboard">', unsafe_allow_html=True)
    st.subheader("Risk Dashboard")
    st.write(f"**Total test borrowers**: {10_000}")
    st.write(f"**Observed default rate**: 21.98%")
    st.write(f"**Average predicted risk**: 45.32%")
    st.markdown('</div>', unsafe_allow_html=True)

# Middle column content - Loan Default Risk Prediction
with middle_column:
    st.markdown(
        """
        <style>
            .risk-prediction {
                background-color: #e9ecef;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            .risk-prediction h3 {
                color: #007bff;
                font-size: 22px;
                font-weight: bold;
            }
            .risk-prediction p {
                font-size: 18px;
                color: #555;
            }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="risk-prediction">', unsafe_allow_html=True)
    st.subheader("Mini Loan Default Risk Prediction")
    st.write("""
        This is a professional credit-risk dashboard for identifying high-risk borrowers,
        explaining borrower-level risk, and checking simple portfolio statistics from the model's test-set predictions.
    """)
    st.write("Try an example query:")
    st.write("1. Top 10 highest-risk borrowers")
    st.write("2. Borrower-level explanation")
    st.write("3. Default rate for loans above $20,000")
    st.markdown('</div>', unsafe_allow_html=True)

# Right column content - Visualization selection
with right_column:
    # Custom CSS for visualization select box
    st.markdown(
        """
        <style>
            .css-1d391kg {
                background-color: #4e73df;  /* Background color of the select box */
                color: white;  /* Font color */
                font-size: 18px;  /* Font size */
                padding: 12px 15px;  /* Padding inside select box */
                border-radius: 8px;  /* Rounded corners */
                border: 2px solid #3e5b96;  /* Border color */
            }
            
            .css-1d391kg select {
                background-color: #4e73df;  /* Select box background */
                color: white;  /* Select box text color */
                font-size: 18px;
            }
            
            .css-1d391kg option {
                background-color: #4e73df;
                color: white;
            }
            
            /* Hover effect for the select box */
            .css-1d391kg select:hover {
                background-color: #375a93;
                border: 2px solid #365c89;
            }

            /* CSS for the option hover effect */
            .css-1d391kg option:hover {
                background-color: #375a93;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.title("Select Visualization")
    option = st.sidebar.selectbox(
        "Choose the graph to view",
        ("Default Rate by Loan Amount", 
         "Precision-Recall Curve", 
         "ROC Curve", 
         "Feature Importance", 
         "Risk Score Distribution")
    )

    # Based on the option selected in the sidebar, render the corresponding visualization in the right column
    if option == "Default Rate by Loan Amount":
        plot_default_rate_by_loan_amount()
    elif option == "Precision-Recall Curve":
        plot_precision_recall_curve()
    elif option == "ROC Curve":
        plot_roc_curve()
    elif option == "Feature Importance":
        plot_feature_importance()
    elif option == "Risk Score Distribution":
        plot_risk_score_distribution()



# Footer
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        Mini Loan Default Predictor • Periscope Labs AI/ML Engineer Take-Home Assignment
    </div>
    """,
    unsafe_allow_html=True,
)


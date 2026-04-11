import streamlit as st
import pandas as pd
import joblib
import json

from src.churn_analysis import compute_churn_risk
from src.feature_engineering import create_ticket_text

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Support Operations Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --ink: #0c1b2a;
        --muted: #64748b;
        --card: rgba(255, 255, 255, 0.78);
        --border: rgba(15, 23, 42, 0.08);
        --brand: #0f766e;
        --brand-soft: #dff8f2;
        --alert: #b91c1c;
        --alert-soft: #fee2e2;
        --bg-1: #f2f7f9;
        --bg-2: #eff6ff;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 10%, rgba(15, 118, 110, 0.14), transparent 35%),
            radial-gradient(circle at 92% 5%, rgba(37, 99, 235, 0.16), transparent 32%),
            linear-gradient(135deg, var(--bg-1), var(--bg-2));
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-shell {
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.65);
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(4px);
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.62));
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
        margin-bottom: 1rem;
        animation: rise-in 0.55s ease-out;
    }

    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.74rem;
        color: #0f766e;
        font-weight: 800;
        font-family: 'Space Grotesk', sans-serif;
    }

    .hero-title {
        margin-top: 0.35rem;
        margin-bottom: 0.45rem;
        font-size: clamp(1.6rem, 2.7vw, 2.4rem);
        line-height: 1.15;
        color: #0f172a;
        font-weight: 800;
        font-family: 'Space Grotesk', sans-serif;
    }

    .hero-sub {
        color: #334155;
        max-width: 70ch;
        margin: 0;
        font-size: 0.98rem;
    }

    .section-title {
        margin-top: 0.35rem;
        margin-bottom: 0.55rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }

    .kpi-card {
        border-radius: 16px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--border);
        background: var(--card);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        animation: rise-in 0.4s ease-out;
    }

    .kpi-card--brand {
        background: linear-gradient(160deg, #ffffff, var(--brand-soft));
    }

    .kpi-card--alert {
        background: linear-gradient(160deg, #ffffff, var(--alert-soft));
    }

    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.35rem;
        font-weight: 700;
    }

    .kpi-value {
        font-size: clamp(1.35rem, 2vw, 1.9rem);
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        margin-bottom: 0.22rem;
    }

    .kpi-sub {
        font-size: 0.83rem;
        color: #475569;
    }

    .panel {
        border-radius: 16px;
        background: var(--card);
        border: 1px solid var(--border);
        padding: 0.85rem 0.95rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }

    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }

    .stTabs [role="tab"] {
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: rgba(255, 255, 255, 0.7);
        padding: 0.45rem 0.95rem;
        font-weight: 700;
        color: #334155;
    }

    .stTabs [aria-selected="true"] {
        background: #0f766e !important;
        color: #ffffff !important;
        border-color: #0f766e !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc, #edf2f7);
        border-right: 1px solid rgba(148, 163, 184, 0.22);
    }

    @keyframes rise-in {
        from {
            opacity: 0;
            transform: translateY(7px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (max-width: 900px) {
        .hero-shell {
            padding: 1.1rem 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load trained model and vectorizer
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/priority_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()


@st.cache_data
def load_model_metrics():
    try:
        with open("models/priority_model_metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


model_metrics = load_model_metrics()

# --------------------------------------------------
# Sidebar upload
# --------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload support tickets CSV",
    type=["csv"]
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data/raw/tickets.csv")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Create combined ticket text
    df = create_ticket_text(df)

    return df


df = load_data(uploaded_file)

# --------------------------------------------------
# Compute churn risk
# --------------------------------------------------
df = compute_churn_risk(df)

# --------------------------------------------------
# Predict ticket priority using saved ML model
# --------------------------------------------------
X_vectorized = vectorizer.transform(df["ticket_text"])
df["predicted_priority"] = model.predict(X_vectorized)

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filters")

priority_filter = st.sidebar.multiselect(
    "Ticket Priority",
    options=sorted(df["ticket_priority"].dropna().unique()),
    default=sorted(df["ticket_priority"].dropna().unique())
)

churn_filter = st.sidebar.multiselect(
    "Churn Risk Level",
    options=["Low", "Medium", "High"],
    default=["Low", "Medium", "High"]
)

filtered_df = df[
    (df["ticket_priority"].isin(priority_filter)) &
    (df["churn_risk"].isin(churn_filter))
]

def kpi_card(title: str, value: str, subtitle: str, tone: str = ""):
    tone_class = f"kpi-card--{tone}" if tone else ""
    st.markdown(
        f"""
        <div class="kpi-card {tone_class}">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Premium hero
st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">Support Intelligence Suite</div>
        <div class="hero-title">Support Operations Command Center</div>
        <p class="hero-sub">
            Monitor ticket pressure, surface at-risk customers, and align escalation decisions
            with live operational and model insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

high_risk_df = filtered_df[filtered_df["churn_risk"] == "High"]
total_tickets = len(filtered_df)
critical_tickets = int((filtered_df["ticket_priority"] == "Critical").sum())
high_risk_customers = int(len(high_risk_df))
high_risk_rate = (high_risk_customers / total_tickets * 100) if total_tickets > 0 else 0.0
test_accuracy = (
    f"{model_metrics['test']['accuracy']:.2%}" if model_metrics is not None else "N/A"
)

st.markdown('<div class="section-title">Executive Snapshot</div>', unsafe_allow_html=True)
snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)

with snap_col1:
    kpi_card("Active Tickets", f"{total_tickets:,}", "Current filtered backlog", "brand")
with snap_col2:
    kpi_card("Critical Tickets", f"{critical_tickets:,}", "Immediate escalation pressure")
with snap_col3:
    kpi_card("High-Risk Customers", f"{high_risk_customers:,}", f"{high_risk_rate:.1f}% of filtered tickets", "alert")
with snap_col4:
    kpi_card("Model Test Accuracy", test_accuracy, "Latest saved priority model run")

tab_overview, tab_risk, tab_model = st.tabs(
    ["Operations Overview", "High-Risk Queue", "Model Quality"]
)

with tab_overview:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Priority Distribution</div>', unsafe_allow_html=True)
        priority_dist = filtered_df["ticket_priority"].value_counts().sort_index()
        if not priority_dist.empty:
            st.bar_chart(priority_dist)
        else:
            st.info("No ticket priority data for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Churn Risk Mix</div>', unsafe_allow_html=True)
        churn_dist = filtered_df["churn_risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        if churn_dist.sum() > 0:
            st.bar_chart(churn_dist)
        else:
            st.info("No churn risk data for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Latest Filtered Tickets</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered_df[
            [
                "customer_email",
                "ticket_priority",
                "predicted_priority",
                "customer_satisfaction_rating",
                "ticket_status",
                "churn_risk",
            ]
        ].head(250),
        use_container_width=True,
        height=360,
    )

with tab_risk:
    st.markdown('<div class="section-title">Customers Requiring Immediate Follow-up</div>', unsafe_allow_html=True)

    if len(high_risk_df) > 0:
        st.dataframe(
            high_risk_df[
                [
                    "customer_email",
                    "ticket_priority",
                    "predicted_priority",
                    "customer_satisfaction_rating",
                    "ticket_status",
                    "churn_risk",
                ]
            ],
            use_container_width=True,
            height=460,
        )

        st.download_button(
            label="Export High-Risk Customers",
            data=high_risk_df.to_csv(index=False),
            file_name="high_risk_customers.csv",
            mime="text/csv",
        )
    else:
        st.info("No customers currently match the selected high-risk filters.")

with tab_model:
    st.markdown('<div class="section-title">Priority Model Evaluation</div>', unsafe_allow_html=True)

    if model_metrics is not None:
        eval_col1, eval_col2, eval_col3 = st.columns(3)
        with eval_col1:
            st.metric("Validation Accuracy", f"{model_metrics['validation']['accuracy']:.2%}")
        with eval_col2:
            st.metric("Test Accuracy", f"{model_metrics['test']['accuracy']:.2%}")
        with eval_col3:
            st.metric("Test Macro F1", f"{model_metrics['test']['macro_f1']:.2%}")

        st.caption(f"Model trained at (UTC): {model_metrics.get('timestamp_utc', 'N/A')}")

        report_df = pd.DataFrame(model_metrics["test"]["classification_report"]).transpose()
        st.dataframe(report_df, use_container_width=True, height=360)
    else:
        st.info(
            "No saved model evaluation report found. Run the training script to generate "
            "models/priority_model_metrics.json."
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption(
    "Churn risk is calculated using transparent business rules "
    "to support decision-making rather than replace human judgment."
)
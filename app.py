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
    :root {
        --ink: #132535;
        --muted: #526174;
        --card: rgba(255, 255, 255, 0.82);
        --border: rgba(19, 37, 53, 0.11);
        --brand: #0b7a75;
        --brand-soft: #d8f7f1;
        --accent: #1d4ed8;
        --warn: #c2410c;
        --warn-soft: #ffedd5;
        --bg-1: #f8f5ef;
        --bg-2: #edf5ff;
    }

    .stApp {
        background:
            radial-gradient(circle at 7% 9%, rgba(11, 122, 117, 0.14), transparent 34%),
            radial-gradient(circle at 92% 6%, rgba(29, 78, 216, 0.14), transparent 32%),
            radial-gradient(circle at 50% 92%, rgba(194, 65, 12, 0.08), transparent 38%),
            linear-gradient(135deg, var(--bg-1), var(--bg-2));
        color: var(--ink);
        font-family: 'Bahnschrift', 'Trebuchet MS', 'Segoe UI', sans-serif;
    }

    /* Override Streamlit theme vars to guarantee readable contrast on light UI */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        --text-color: #132535;
        --primary-color: #0b7a75;
        --secondary-background-color: #f4f7fb;
        --background-color: transparent;
    }

    .stApp,
    .stApp p,
    .stApp li,
    .stApp span,
    .stApp label,
    .stApp small,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    div[data-testid="stMarkdownContainer"] * {
        color: var(--ink);
    }

    div[data-testid="stCaptionContainer"] p,
    .stCaption {
        color: var(--muted) !important;
    }

    .stApp a {
        color: var(--accent) !important;
    }

    [data-testid="stSidebar"] a {
        color: #1d4ed8 !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero-shell {
        border-radius: 22px;
        border: 1px solid rgba(255, 255, 255, 0.75);
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(6px);
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.70));
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12);
        margin-bottom: 1rem;
        animation: rise-in 0.55s ease-out;
    }

    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.74rem;
        color: var(--brand);
        font-weight: 800;
    }

    .hero-title {
        margin-top: 0.35rem;
        margin-bottom: 0.45rem;
        font-size: clamp(1.6rem, 2.7vw, 2.4rem);
        line-height: 1.15;
        color: #102033;
        font-weight: 800;
        letter-spacing: -0.02em;
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
        font-size: 1.1rem;
        font-weight: 800;
        color: #102033;
    }

    .kpi-card {
        border-radius: 16px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--border);
        background: var(--card);
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
        animation: rise-in 0.4s ease-out;
    }

    .kpi-card--brand {
        background: linear-gradient(160deg, #ffffff, var(--brand-soft));
    }

    .kpi-card--alert {
        background: linear-gradient(160deg, #ffffff, var(--warn-soft));
    }

    .kpi-card--accent {
        background: linear-gradient(160deg, #ffffff, #dbeafe);
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
        color: #102033;
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
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.6rem 0.8rem;
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetricValue"] {
        color: #102033 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #14532d !important;
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
        background: var(--brand) !important;
        color: #ffffff !important;
        border-color: var(--brand) !important;
    }

    .stTabs [role="tabpanel"] * {
        color: #132535;
    }

    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--brand), var(--accent));
    }

    div.stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(11, 122, 117, 0.4);
        background: linear-gradient(120deg, #ffffff, #d8f7f1);
        color: #083344;
        font-weight: 700;
    }

    div.stDownloadButton > button:hover {
        border-color: rgba(11, 122, 117, 0.7);
        color: #042f2e;
    }

    .stButton > button {
        color: #102033 !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] div,
    div[data-baseweb="tag"] span {
        color: #102033 !important;
    }

    div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.72);
    }

    /* BaseWeb dropdown popup can render outside .stApp; force readable options */
    div[role="listbox"],
    div[role="option"],
    ul[role="listbox"] li {
        color: #102033 !important;
        background: #ffffff !important;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] * {
        color: #102033 !important;
    }

    div[data-testid="stAlert"] *,
    div[data-testid="stNotification"] *,
    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] * {
        color: #102033 !important;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploaderDropzone"] * {
        color: #102033 !important;
    }

    [data-testid="stWidgetLabel"] *,
    .stSelectbox label,
    .stMultiSelect label,
    .stCheckbox label,
    .stRadio label,
    .stSlider label {
        color: #102033 !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc, #edf2f7);
        border-right: 1px solid rgba(148, 163, 184, 0.22);
    }

    section[data-testid="stSidebar"] * {
        color: #102033 !important;
    }

    /* Dark mode safety: enforce light text and dark surfaces everywhere */
    @media (prefers-color-scheme: dark) {
        :root {
            --ink: #e6eef8;
            --muted: #b8c5d7;
            --card: rgba(14, 23, 34, 0.88);
            --border: rgba(148, 163, 184, 0.28);
            --brand: #2dd4bf;
            --brand-soft: #0f2f32;
            --accent: #60a5fa;
            --warn-soft: #3b1f15;
            --bg-1: #070c14;
            --bg-2: #0d1523;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            --text-color: #e6eef8;
            --primary-color: #2dd4bf;
            --secondary-background-color: #0f1724;
            --background-color: transparent;
            background:
                radial-gradient(circle at 7% 9%, rgba(45, 212, 191, 0.18), transparent 32%),
                radial-gradient(circle at 92% 6%, rgba(96, 165, 250, 0.16), transparent 30%),
                linear-gradient(135deg, #070c14, #0d1523) !important;
            color: #e6eef8 !important;
        }

        .stApp,
        .stApp p,
        .stApp li,
        .stApp span,
        .stApp label,
        .stApp small,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        div[data-testid="stMarkdownContainer"] * {
            color: #e6eef8 !important;
        }

        .hero-shell,
        .panel,
        .kpi-card,
        div[data-testid="stMetric"],
        details[data-testid="stExpander"],
        div[data-testid="stAlert"],
        div[data-testid="stNotification"] {
            background: rgba(14, 23, 34, 0.88) !important;
            border-color: rgba(148, 163, 184, 0.28) !important;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35) !important;
            color: #e6eef8 !important;
        }

        .hero-title,
        .hero-sub,
        .section-title,
        .kpi-value,
        .kpi-sub,
        .kpi-label,
        div[data-testid="stMetric"] label,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricDelta"],
        div[data-testid="stCaptionContainer"] p,
        .stCaption {
            color: #e6eef8 !important;
        }

        .stTabs [role="tab"] {
            background: rgba(15, 23, 36, 0.86) !important;
            border-color: rgba(148, 163, 184, 0.35) !important;
            color: #dbe7f6 !important;
        }

        .stTabs [aria-selected="true"] {
            background: #2dd4bf !important;
            border-color: #2dd4bf !important;
            color: #042a2a !important;
        }

        .stTabs [role="tabpanel"] * {
            color: #e6eef8 !important;
        }

        div[data-baseweb="input"],
        div[data-baseweb="input"] input,
        div[data-baseweb="select"],
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] div,
        div[data-baseweb="tag"] {
            background: #111a28 !important;
            color: #e6eef8 !important;
            border-color: rgba(148, 163, 184, 0.35) !important;
        }

        div[role="listbox"],
        div[role="option"],
        ul[role="listbox"] li {
            background: #111a28 !important;
            color: #e6eef8 !important;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] * {
            color: #e6eef8 !important;
            background-color: transparent;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1320, #0f1827) !important;
            border-right: 1px solid rgba(148, 163, 184, 0.3);
        }

        section[data-testid="stSidebar"] *,
        [data-testid="stSidebar"] a,
        [data-testid="stWidgetLabel"] *,
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploaderDropzone"] * {
            color: #e6eef8 !important;
        }

        .stButton > button,
        div.stDownloadButton > button {
            background: #102436 !important;
            border-color: rgba(148, 163, 184, 0.35) !important;
            color: #e6eef8 !important;
        }

        .stApp a,
        [data-testid="stSidebar"] a {
            color: #7dd3fc !important;
        }
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
operations_health = max(0.0, 100.0 - (high_risk_rate * 1.5))

st.markdown('<div class="section-title">Executive Snapshot</div>', unsafe_allow_html=True)
snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)

with snap_col1:
    kpi_card("Active Tickets", f"{total_tickets:,}", "Current filtered backlog", "brand")
with snap_col2:
    kpi_card("Critical Tickets", f"{critical_tickets:,}", "Immediate escalation pressure")
with snap_col3:
    kpi_card("High-Risk Customers", f"{high_risk_customers:,}", f"{high_risk_rate:.1f}% of filtered tickets", "alert")
with snap_col4:
    kpi_card("Model Test Accuracy", test_accuracy, "Latest saved priority model run", "accent")

st.markdown('<div class="panel">', unsafe_allow_html=True)
health_col1, health_col2 = st.columns([2, 1])
with health_col1:
    st.markdown('<div class="section-title">Operations Health Index</div>', unsafe_allow_html=True)
    st.progress(int(operations_health))
    st.caption("Index combines current high-risk concentration and ticket pressure for quick triage confidence.")
with health_col2:
    st.metric("Health Score", f"{operations_health:.0f}/100")
st.markdown('</div>', unsafe_allow_html=True)

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

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Ticket Status Distribution</div>', unsafe_allow_html=True)
        status_dist = filtered_df["ticket_status"].fillna("Unknown").value_counts().sort_values(ascending=False)
        if not status_dist.empty:
            st.bar_chart(status_dist)
        else:
            st.info("No ticket status data for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with status_col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Priority vs Predicted Match Rate</div>', unsafe_allow_html=True)
        if total_tickets > 0:
            match_rate = (filtered_df["ticket_priority"] == filtered_df["predicted_priority"]).mean() * 100
            st.metric("Current Match Rate", f"{match_rate:.1f}%")
            st.progress(int(match_rate))
            st.caption("Share of filtered tickets where assigned and predicted priority are identical.")
        else:
            st.info("No tickets available for selected filters.")
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
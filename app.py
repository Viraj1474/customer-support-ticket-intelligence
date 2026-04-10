import streamlit as st
import pandas as pd
import joblib

from src.churn_analysis import compute_churn_risk
from src.feature_engineering import create_ticket_text

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Support Operations Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Title and description
# --------------------------------------------------
st.title("Support Operations Dashboard")

st.markdown("""
This dashboard supports customer service and operations teams in
monitoring active tickets, identifying urgent issues, and detecting
customers who may require proactive follow-up.

The focus of this system is operational visibility and decision support.
""")

# --------------------------------------------------
# Load trained model and vectorizer
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/priority_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()

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

# --------------------------------------------------
# KPI metrics
# --------------------------------------------------
st.subheader("Current Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Tickets", len(filtered_df))

with col2:
    st.metric(
        "Critical Tickets",
        (filtered_df["ticket_priority"] == "Critical").sum()
    )

with col3:
    st.metric(
        "High-Risk Customers",
        (filtered_df["churn_risk"] == "High").sum()
    )

# --------------------------------------------------
# High-risk customer section
# --------------------------------------------------
st.subheader("Customers Requiring Immediate Follow-up")

high_risk_df = filtered_df[
    filtered_df["churn_risk"] == "High"
]

if len(high_risk_df) > 0:
    st.dataframe(
        high_risk_df[
            [
                "customer_email",
                "ticket_priority",
                "predicted_priority",
                "customer_satisfaction_rating",
                "ticket_status",
                "churn_risk"
            ]
        ],
        use_container_width=True
    )

    st.download_button(
        label="Export High-Risk Customers",
        data=high_risk_df.to_csv(index=False),
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )
else:
    st.info("No customers currently match the selected high-risk filters.")

# --------------------------------------------------
# Full filtered ticket table
# --------------------------------------------------
st.subheader("All Filtered Tickets")

st.dataframe(
    filtered_df[
        [
            "customer_email",
            "ticket_priority",
            "predicted_priority",
            "customer_satisfaction_rating",
            "ticket_status",
            "churn_risk"
        ]
    ],
    use_container_width=True
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption(
    "Churn risk is calculated using transparent business rules "
    "to support decision-making rather than replace human judgment."
)
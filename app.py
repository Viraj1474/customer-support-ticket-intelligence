import streamlit as st
import pandas as pd
from src.churn_analysis import compute_churn_risk

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Support Operations Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Title and context
# --------------------------------------------------
st.title("Support Operations Dashboard")

st.markdown("""
This dashboard supports customer service and operations teams in
monitoring active tickets, identifying urgent issues, and detecting
customers who may require proactive follow-up.

The focus of this system is operational visibility and decision support.
""")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/tickets.csv")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    from src.feature_engineering import create_ticket_text
    df = create_ticket_text(df)


    return df


df = load_data()
df = compute_churn_risk(df)

# --------------------------------------------------
# Recreate churn logic using business rules
# --------------------------------------------------

# Convert date columns
df['date_of_purchase'] = pd.to_datetime(df['date_of_purchase'], errors='coerce')
df['time_to_resolution'] = pd.to_datetime(df['time_to_resolution'], errors='coerce')

# Calculate resolution time in hours
df['resolution_time_hours'] = (
    df['time_to_resolution'] - df['date_of_purchase']
).dt.total_seconds() / 3600

# Cap unrealistic values
df['resolution_time_hours_capped'] = df['resolution_time_hours'].clip(upper=2160)

# Business indicators
df['low_satisfaction'] = df['customer_satisfaction_rating'] <= 2
df['high_priority_flag'] = df['ticket_priority'].isin(['High', 'Critical'])
df['long_resolution_flag'] = df['resolution_time_hours_capped'] > 72

ticket_counts = df.groupby('customer_email').size()
df['repeat_customer_flag'] = df['customer_email'].map(ticket_counts) > 3

# Churn score calculation
df['churn_score'] = (
    df['low_satisfaction'].fillna(False).astype(int) +
    df['high_priority_flag'].astype(int) +
    df['long_resolution_flag'].astype(int) +
    df['repeat_customer_flag'].astype(int)
)

def churn_label(score):
    if score >= 3:
        return "High"
    elif score == 2:
        return "Medium"
    else:
        return "Low"

df['churn_risk'] = df['churn_score'].apply(churn_label)

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filters")

priority_filter = st.sidebar.multiselect(
    "Ticket priority",
    options=df['ticket_priority'].unique(),
    default=df['ticket_priority'].unique()
)

churn_filter = st.sidebar.multiselect(
    "Churn risk level",
    options=df['churn_risk'].unique(),
    default=df['churn_risk'].unique()
)

filtered_df = df[
    (df['ticket_priority'].isin(priority_filter)) &
    (df['churn_risk'].isin(churn_filter))
]

# --------------------------------------------------
# Key metrics
# --------------------------------------------------
st.markdown("### Current Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Active tickets",
    len(filtered_df)
)

col2.metric(
    "Critical tickets requiring attention",
    (filtered_df['ticket_priority'] == 'Critical').sum()
)

col3.metric(
    "Customers at high churn risk",
    (filtered_df['churn_risk'] == 'High').sum()
)

# --------------------------------------------------
# High churn risk customers
# --------------------------------------------------
st.markdown("### Customers Requiring Immediate Follow-up")

st.markdown("""
The following customers show multiple indicators of dissatisfaction,
including repeated support requests, long resolution times, or
high-priority issues. These cases should be reviewed proactively.
""")

high_risk_df = filtered_df[filtered_df['churn_risk'] == 'High']

st.dataframe(
    high_risk_df[
        [
            'customer_email',
            'ticket_priority',
            'customer_satisfaction_rating',
            'ticket_status'
        ]
    ],
    use_container_width=True
)

# --------------------------------------------------
# Export option
# --------------------------------------------------
st.download_button(
    label="Export customers requiring follow-up",
    data=high_risk_df.to_csv(index=False),
    file_name="high_churn_customers.csv",
    mime="text/csv"
)

# --------------------------------------------------
# Footer note
# --------------------------------------------------
st.caption(
    "Churn risk is calculated using transparent operational rules rather "
    "than opaque models to ensure clarity and trust for support teams."
)

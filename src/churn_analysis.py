import pandas as pd

def compute_churn_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute churn risk using transparent business rules.
    """

    df = df.copy()

    # Convert dates
    df['date_of_purchase'] = pd.to_datetime(df['date_of_purchase'], errors='coerce')
    df['time_to_resolution'] = pd.to_datetime(df['time_to_resolution'], errors='coerce')

    # Resolution time in hours
    df['resolution_time_hours'] = (
        df['time_to_resolution'] - df['date_of_purchase']
    ).dt.total_seconds() / 3600

    df['resolution_time_hours_capped'] = df['resolution_time_hours'].clip(upper=2160)

    # Churn signals
    df['low_satisfaction'] = df['customer_satisfaction_rating'] <= 2
    df['high_priority_flag'] = df['ticket_priority'].isin(['High', 'Critical'])
    df['long_resolution_flag'] = df['resolution_time_hours_capped'] > 72

    ticket_counts = df.groupby('customer_email').size()
    df['repeat_customer_flag'] = df['customer_email'].map(ticket_counts) > 3

    # Score
    df['churn_score'] = (
        df['low_satisfaction'].fillna(False).astype(int)
        + df['high_priority_flag'].astype(int)
        + df['long_resolution_flag'].astype(int)
        + df['repeat_customer_flag'].astype(int)
    )

    def label(score):
        if score >= 3:
            return "High"
        elif score == 2:
            return "Medium"
        return "Low"

    df['churn_risk'] = df['churn_score'].apply(label)

    return df

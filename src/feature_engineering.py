import pandas as pd

def create_ticket_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine ticket subject and description into a single text field.
    """
    df = df.copy()

    df['ticket_text'] = (
        df['ticket_subject'].fillna('') + ' ' +
        df['ticket_description'].fillna('')
    )

    return df


def create_resolution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create resolution time features used for analytics and churn logic.
    """
    df = df.copy()

    df['date_of_purchase'] = pd.to_datetime(df['date_of_purchase'], errors='coerce')
    df['time_to_resolution'] = pd.to_datetime(df['time_to_resolution'], errors='coerce')

    df['resolution_time_hours'] = (
        df['time_to_resolution'] - df['date_of_purchase']
    ).dt.total_seconds() / 3600

    df['resolution_time_hours_capped'] = df['resolution_time_hours'].clip(upper=2160)

    return df

import pandas as pd
import joblib
import json
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.feature_engineering import create_ticket_text

def evaluate_split(y_true, y_pred, labels):
    """
    Build a metrics dictionary for one data split.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# Load dataset
df = pd.read_csv("data/raw/tickets.csv")

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Create combined text column
df = create_ticket_text(df)

# Features and labels
X = df["ticket_text"]
y = df["ticket_priority"]

# Split into train/validation/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

# Convert text to vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Train model on train split only
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Evaluate on validation and test splits
y_val_pred = model.predict(X_val_vectorized)
y_test_pred = model.predict(X_test_vectorized)

labels = sorted(y.unique().tolist())
metrics_payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "splits": {
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
    },
    "class_labels": labels,
    "validation": evaluate_split(y_val, y_val_pred, labels),
    "test": evaluate_split(y_test, y_test_pred, labels),
}

# Save trained files and metrics report
joblib.dump(model, "models/priority_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

with open("models/priority_model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2)

print("Model, vectorizer, and evaluation metrics saved successfully.")
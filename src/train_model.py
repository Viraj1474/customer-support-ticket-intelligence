import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import create_ticket_text

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

# Convert text to vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

# Save trained files
joblib.dump(model, "models/priority_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model and vectorizer saved successfully.")
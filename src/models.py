from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }

    return model, metrics

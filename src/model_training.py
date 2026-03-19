import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

def train_logistic_regression(X_train, y_train, max_iter=100, class_weight="balanced", random_state=42):
    """
    Train Logistic Regression model
    """
    model = LogisticRegression(max_iter=max_iter, class_weight=class_weight, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, class_weight="balanced", random_state=42):
    """
    Train Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("Accuracy:", accuracy)
    print("AUC:", roc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return accuracy, roc

def save_model(model, filename="saved_models/best_model.pkl"):
    """
    Save trained model
    """
    joblib.dump(model, filename)
    print("Model saved in", filename)
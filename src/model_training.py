import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

def train_models_with_gridsearch(X_train, y_train):
    """
    Train multiple models using Pipeline and GridSearchCV
    """
    # Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",LogisticRegression())
    ])

    # Parameter grid for different models
    param_grid = [
        {
            "model": [LogisticRegression(class_weight="balanced", random_state=42)],
            "model__C": [0.01, 0.1, 1, 10]
        },
        {
            "model":[RandomForestClassifier(class_weight="balanced", random_state=42)],
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        }
    ]

    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross Validation AUC:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_

    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print("\nModel Evaluation Results")
    print("-------------------------")
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
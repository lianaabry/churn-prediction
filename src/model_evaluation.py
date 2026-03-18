import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

def print_classification_metrics(model, X_test, y_test):
    """
    Print classification matrix
    """
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix
    """
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 4))

    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], '--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.show()

def plot_feature_importance(model, X_train):
    """
    Plot feature importance (for tree models)
    """

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_
        features = X_train.columns

        sns.barplot(x=importance, y=features)

        plt.title("Feature Importance")

        plt.show()

    else:
        print("Model does not support feature importance.")
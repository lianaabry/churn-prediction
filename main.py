from src.data_preprocessing import *
from src.feature_engineering import *
from src.model_training import *
from src.model_evaluation import *

df = load_data("data/churn.csv")

df = clean_data(df)

df = feature_engineering_pipeline(df)

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_data(X, y)

# Train models
lr_model = train_logistic_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

print("Logistic Regression Results")
evaluate_model(lr_model, X_test, y_test)

print("Random Forest Results")
evaluate_model(rf_model, X_test, y_test)

# Save best model
save_model(rf_model)
save_model(lr_model)


print_classification_metrics(rf_model, X_test, y_test)

plot_confusion_matrix(rf_model, X_test, y_test)

plot_roc_curve(rf_model, X_test, y_test)

plot_feature_importance(rf_model, X_train)

print_classification_metrics(lr_model, X_test, y_test)

plot_confusion_matrix(lr_model, X_test, y_test)

plot_roc_curve(lr_model, X_test, y_test)

plot_feature_importance(lr_model, X_train)
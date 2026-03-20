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
best_model = train_models_with_gridsearch(X_train, y_train)


print("Best model's results")
evaluate_model(best_model, X_test, y_test)

# Save best model
print("Best model saved:")
save_model(best_model)

plot_confusion_matrix(best_model, X_test, y_test)

plot_roc_curve(best_model, X_test, y_test)



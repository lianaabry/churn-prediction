import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def run_shap_analysis(model_path, X_test):
    """
    Generate SHAP explainability plots
    """

    # Load trained model
    pipeline = joblib.load(model_path)
    model = pipeline.named_steps["model"]

    if "Exited" in X_test.columns:
        X_test = X_test.drop("Exited", axis=1)

    # Scale the data using pipeline scaler
    X_scaled  = pipeline.named_steps["scaler"].transform(X_test)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_scaled)

    # Summary plot
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values[1], X_scaled, feature_names=X_test.columns, show=False)
    plt.savefig("plots/shap_summary.png", dpi=300)
    plt.close()

    # Feature importance bar plot
    print("Generating SHAP summary plot bar...")
    shap.summary_plot(shap_values[1], X_scaled, feature_names=X_test.columns, plot_type="bar", show=False)
    plt.savefig("plots/shap_summary_bar.png", dpi=300)
    plt.close()

    # Explain one prediction
    print("Generating SHAP force plot for first customer...")


    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[1][0],
            base_values=explainer.expected_value[1],
            data=X_scaled[0],
            feature_names=X_test.columns,


        ), show=False, max_display=10
    )
    plt.tight_layout()
    plt.savefig("plots/shap_summary_force.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.close()

X_test = pd.read_csv("data/test.csv")
run_shap_analysis("saved_models/best_model.pkl", X_test)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

def explain_model():
    processed_dir = os.path.join("dataset", "processed")
    images_dir = os.path.join("reports", "images")
    models_dir = "models"
    os.makedirs(images_dir, exist_ok=True)

    print("Loading best model and test dataset...")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model not found. Run train.py first.")
        
    model = joblib.load(best_model_path)
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))

    print("Initializing SHAP explainer...")
    # Determine the model type to use the most efficient SHAP explainer
    model_type = type(model).__name__
    print(f"Model type: {model_type}")

    try:
        if "XGB" in model_type or "RandomForest" in model_type or "DecisionTree" in model_type:
            explainer = shap.TreeExplainer(model)
            # In SHAP, TreeExplainer on RandomForest/DecisionTree might return list for multi-class/binary.
            # For XGBoost it returns a single array or matrix. Let's handle shape.
            shap_values = explainer.shap_values(X_test)
            
            # If shap_values is list (typical for RF in binary classification), select index 1 (positive class)
            if isinstance(shap_values, list):
                # Random Forest binary class explanation: list of 2 arrays (negative, positive)
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # Some versions return shape (n_samples, n_features, n_classes)
                shap_values = shap_values[:, :, 1]
        else:
            # Fallback to general explainer
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
    except Exception as e:
        print(f"Tree/General Explainer failed, falling back to KernelExplainer: {e}")
        # Sample background data for kernel explainer to make it fast
        background = shap.kmeans(X_test, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    
    # Save global SHAP summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Global Feature Importance & Impact Plot", fontsize=14, pad=15)
    
    shap_plot_path = os.path.join(images_dir, "shap_summary.png")
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot successfully saved to {shap_plot_path}!")

    # Calculate mean absolute SHAP values to rank and display feature importance numerically
    if isinstance(shap_values, np.ndarray):
        mean_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Mean_Abs_SHAP": mean_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False)
        
        print("\nTop 10 Most Important Features (SHAP):")
        print(importance_df.head(10).to_string(index=False))
        
        # Save importance text report
        importance_df.to_csv(os.path.join(processed_dir, "shap_feature_importance.csv"), index=False)
        print("Feature importance list saved to dataset/processed/shap_feature_importance.csv")

if __name__ == "__main__":
    explain_model()

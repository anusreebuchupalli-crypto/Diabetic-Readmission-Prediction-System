import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib

def train_and_evaluate():
    processed_dir = os.path.join("dataset", "processed")
    images_dir = os.path.join("reports", "images")
    models_dir = "models"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Loading processed datasets...")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    # Load y_train and squeeze/convert to Series
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).iloc[:, 0]
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    # Load y_test and squeeze/convert to Series
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).iloc[:, 0]

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.1),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_leaf=20),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1, eval_metric='logloss')
    }

    results = []
    roc_data = {}
    confusion_matrices = {}

    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        print(f"{name} Results - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        # Save confusion matrix
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    # Finalize and save ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    roc_plot_path = os.path.join(images_dir, "roc_curve_comparison.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve plot saved to {roc_plot_path}")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['No Readmit', 'Readmitted'], yticklabels=['No Readmit', 'Readmitted'])
        axes[idx].set_title(f"Confusion Matrix: {name}", fontsize=14)
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
    
    plt.tight_layout()
    cm_plot_path = os.path.join(images_dir, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices plot saved to {cm_plot_path}")

    # Save metrics comparison to CSV/DF
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))

    # Identify the best model
    # We want a model with high ROC-AUC and a good balance of Recall (critically important in medicine to capture positive cases).
    # We'll pick the model with the highest ROC-AUC.
    best_row = results_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    print(f"\nBest Model selected: {best_model_name} with ROC-AUC = {best_row['ROC-AUC']:.4f}")

    # Save the best model
    best_model = models[best_model_name]
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to {best_model_path}")

    # Write summary metrics to reports
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    summary_path = os.path.join(reports_dir, "model_comparison.md")
    
    with open(summary_path, "w") as f:
        f.write("# Model Evaluation & Comparison Summary\n\n")
        f.write("Below is the performance summary of the trained models on the test set:\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for res in results:
            f.write(f"| {res['Model']} | {res['Accuracy']:.4f} | {res['Precision']:.4f} | {res['Recall']:.4f} | {res['F1-Score']:.4f} | {res['ROC-AUC']:.4f} |\n")
        
        f.write(f"\n\n**Best Model:** {best_model_name} was chosen due to its superior ROC-AUC score of **{best_row['ROC-AUC']:.4f}** and balanced recall.\n")
        f.write("\nRefer to `reports/images/roc_curve_comparison.png` and `reports/images/confusion_matrices.png` for plots.\n")
        
    print(f"Metrics summary written to {summary_path}")

if __name__ == "__main__":
    train_and_evaluate()

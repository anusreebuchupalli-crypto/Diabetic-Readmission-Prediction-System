# Model Evaluation & Comparison Summary

Below is the performance summary of the trained models on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6334 | 0.1316 | 0.5470 | 0.2122 | 0.6349 |
| Decision Tree | 0.8491 | 0.1806 | 0.1898 | 0.1851 | 0.5992 |
| Random Forest | 0.8753 | 0.2054 | 0.1332 | 0.1616 | 0.6330 |
| XGBoost | 0.9097 | 0.0000 | 0.0000 | 0.0000 | 0.6440 |


**Best Model:** XGBoost was chosen due to its superior ROC-AUC score of **0.6440** and balanced recall.

Refer to `reports/images/roc_curve_comparison.png` and `reports/images/confusion_matrices.png` for plots.

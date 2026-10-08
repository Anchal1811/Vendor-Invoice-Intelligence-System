"""Evaluation helpers: class-1 metrics, threshold tuning, permutation importance."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_proba_1(model, X):
    return model.predict_proba(X)[:, 1]


def find_best_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Threshold maximizing F-beta (beta>1 favors recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    denom = beta**2 * precision + recall
    fbeta = np.divide(
        (1 + beta**2) * precision * recall, denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    return float(thresholds[np.argmax(fbeta)])


def evaluate_model(model, X_test, y_test, model_name: str, threshold: float = 0.5) -> dict:
    proba = predict_proba_1(model, X_test)
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "threshold": round(threshold, 3),
        "accuracy": (y_pred == y_test).mean(),
        "precision_flagged": precision_score(y_test, y_pred, zero_division=0),
        "recall_flagged": recall_score(y_test, y_pred),
        "f1_flagged": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }

    print(f"--- {model_name} (threshold={threshold:.3f}) ---")
    for k, v in metrics.items():
        if k not in ("model", "threshold"):
            print(f"{k:18s}: {v:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))
    return metrics


def get_feature_importance(model, X_test, y_test, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(
        model, X_test, y_test,
        scoring="f1", n_repeats=n_repeats, random_state=random_state, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
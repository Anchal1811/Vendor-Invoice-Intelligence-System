"""Train and evaluate invoice flagging models.

Usage (run from the project root):
    python invoice_flagging\\train.py --db "C:\\Users\\DELL\\Machine Learning  DS\\Data\\inventory.db"
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import DB_PATH, FEATURES, prepare_dataset, split_data
from model_evaluation import evaluate_model, find_best_threshold, get_feature_importance

RANDOM_STATE = 42
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "models" / "predict_flag_invoice.pkl"


def build_candidates():
    """Model -> (pipeline, param grid). Scaling only where it matters."""
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    logreg_grid = {"clf__C": [0.01, 0.1, 1, 10]}

    rf = Pipeline([
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    }
    return {
        "Logistic Regression": (logreg, logreg_grid),
        "Random Forest": (rf, rf_grid),
    }


def main(db_path: str, output):
    df = prepare_dataset(db_path)
    print(f"Rows: {len(df)} | Flag rate: {df['flag_invoice'].mean():.2%}")
    print(f"Features: {FEATURES}\n")

    X_train, X_test, y_train, y_test = split_data(df, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted = [], {}
    for name, (pipe, grid) in build_candidates().items():
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        print(f"{name}: best CV F1={search.best_score_:.4f} | {search.best_params_}")

        # Threshold chosen on out-of-fold train predictions (no test leakage)
        oof = cross_val_predict(search.best_estimator_, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = find_best_threshold(y_train, oof, beta=2.0)

        metrics = evaluate_model(search.best_estimator_, X_test, y_test, name, threshold)
        metrics["cv_f1"] = search.best_score_
        results.append(metrics)
        fitted[name] = (search.best_estimator_, threshold)

    summary = pd.DataFrame(results).set_index("model")
    print("\n=== Summary (test set) ===")
    print(summary.round(4))

    best_name = summary["f1_flagged"].idxmax()
    best_model, best_threshold = fitted[best_name]
    print(f"\nBest model: {best_name}")
    print(get_feature_importance(best_model, X_test, y_test).round(4))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best_model, "threshold": best_threshold, "features": FEATURES},
        output,
    )
    print(f"Best model saved: {best_name}")
    print(f"Model path: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    main(args.db, args.output)
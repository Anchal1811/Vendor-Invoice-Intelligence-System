from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data


def train_models(x_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest Regression": RandomForestRegressor(max_depth=4, random_state=42),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def evaluate_models(models, x_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(x_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds) * 100,
        }
        print(f"\n{name} Performance:")
        print(f"MAE  : {results[name]['mae']:.2f}")
        print(f"RMSE : {results[name]['rmse']:.2f}")
        print(f"R²   : {results[name]['r2']:.2f}%")
    return results


def save_best_model(models, results):
    best_name = min(results, key=lambda n: results[n]["rmse"])

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "predict_freight_model.pkl"
    joblib.dump(models[best_name], model_path)

    print(f"\nBest model saved: {best_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    df = load_vendor_invoice_data()
    X, y = prepare_features(df)
    x_train, x_test, y_train, y_test = split_data(X, y)

    models = train_models(x_train, y_train)
    results = evaluate_models(models, x_test, y_test)
    save_best_model(models, results)
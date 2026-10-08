import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data


def evaluate_model(model, x_test, y_test, model_name):
    preds = model.predict(x_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds) * 100

    print(f"\n{model_name} Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}%")

    return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2}


def plot_predictions(model, x_test, y_test):
    plt.scatter(x_test, y_test)
    plt.plot(x_test, model.predict(x_test), color="red")
    plt.xlabel("Dollars")
    plt.ylabel("Freight")
    plt.show()


if __name__ == "__main__":
    df = load_vendor_invoice_data()
    X, y = prepare_features(df)
    x_train, x_test, y_train, y_test = split_data(X, y)

    model_files = {
        "Linear Regression": "linear_regression.pkl",
        "Decision Tree Regression": "decision_tree.pkl",
        "Random Forest Regression": "random_forest.pkl",
    }
    for name, file in model_files.items():
        model = joblib.load(file)
        evaluate_model(model, x_test, y_test, name)
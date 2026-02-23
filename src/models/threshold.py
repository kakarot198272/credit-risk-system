import numpy as np
import pandas as pd

MARGIN = 0.10
LGD = 0.60


def calculate_profit(pred_probs, threshold):
    decisions = pred_probs < threshold

    profits = np.where(
        decisions,
        (1 - pred_probs) * MARGIN - pred_probs * LGD,
        0
    )

    return profits.sum()


def evaluate_model(file_path):
    df = pd.read_csv(file_path)
    pred_probs = df["pred_prob"].values

    thresholds = np.linspace(0.01, 0.5, 100)

    best_threshold = 0
    best_profit = -np.inf

    for t in thresholds:
        profit = calculate_profit(pred_probs, t)
        if profit > best_profit:
            best_profit = profit
            best_threshold = t

    return best_threshold, best_profit


if __name__ == "__main__":
    models = {
    "Logistic": "artifacts/metrics/validation_predictions_logistic.csv",
    "LightGBM": "artifacts/metrics/validation_predictions_lightgbm.csv",
    "LightGBM Calibrated": "artifacts/metrics/validation_predictions_lightgbm_calibrated.csv",
    "XGBoost": "artifacts/metrics/validation_predictions_xgboost.csv"
    }



    for name, path in models.items():
        threshold, profit = evaluate_model(path)
        print(f"\n{name}")
        print(f"Optimal Threshold: {threshold:.4f}")
        print(f"Maximum Expected Profit: {profit:.4f}")

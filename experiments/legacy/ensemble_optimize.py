import numpy as np
import pandas as pd

MARGIN = 0.10
LGD = 0.60

LGB_PATH = "artifacts/metrics/validation_predictions_lightgbm.csv"
XGB_PATH = "artifacts/metrics/validation_predictions_xgboost.csv"


def calculate_profit(pred_probs, threshold):
    decisions = pred_probs < threshold

    profits = np.where(
        decisions,
        (1 - pred_probs) * MARGIN - pred_probs * LGD,
        0
    )

    return profits.sum()


def optimize_threshold(pred_probs):
    thresholds = np.linspace(0.01, 0.5, 100)

    best_threshold = 0
    best_profit = -np.inf

    for t in thresholds:
        profit = calculate_profit(pred_probs, t)
        if profit > best_profit:
            best_profit = profit
            best_threshold = t

    return best_threshold, best_profit


def main():
    lgb_df = pd.read_csv(LGB_PATH)
    xgb_df = pd.read_csv(XGB_PATH)

    lgb_probs = lgb_df["pred_prob"].values
    xgb_probs = xgb_df["pred_prob"].values

    best_weight = 0
    best_profit = -np.inf
    best_threshold = 0

    # Try weights from 0 to 1
    weights = np.linspace(0, 1, 21)

    for w in weights:
        ensemble_probs = w * xgb_probs + (1 - w) * lgb_probs

        threshold, profit = optimize_threshold(ensemble_probs)

        if profit > best_profit:
            best_profit = profit
            best_weight = w
            best_threshold = threshold

    print("===== Optimized Ensemble =====")
    print(f"Best Weight for XGBoost: {best_weight:.2f}")
    print(f"Best Weight for LightGBM: {1 - best_weight:.2f}")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Maximum Expected Profit: {best_profit:.4f}")


if __name__ == "__main__":
    main()

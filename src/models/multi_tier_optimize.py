import numpy as np
import pandas as pd

XGB_PATH = "artifacts/metrics/validation_predictions_xgboost.csv"

LGD = 0.60

TIER1_MARGIN = 0.10
TIER2_MARGIN = 0.18


def calculate_multi_tier_profit(pred_probs, t1, t2):
    profits = np.zeros_like(pred_probs)

    # Tier 1
    tier1_mask = pred_probs < t1
    profits[tier1_mask] = (
        (1 - pred_probs[tier1_mask]) * TIER1_MARGIN
        - pred_probs[tier1_mask] * LGD
    )

    # Tier 2
    tier2_mask = (pred_probs >= t1) & (pred_probs < t2)
    profits[tier2_mask] = (
        (1 - pred_probs[tier2_mask]) * TIER2_MARGIN
        - pred_probs[tier2_mask] * LGD
    )

    # Tier 3 → reject → profit = 0

    return profits.sum()


def main():
    df = pd.read_csv(XGB_PATH)
    probs = df["pred_prob"].values

    best_profit = -np.inf
    best_t1 = 0
    best_t2 = 0

    thresholds = np.linspace(0.01, 0.5, 50)

    for t1 in thresholds:
        for t2 in thresholds:
            if t2 <= t1:
                continue

            profit = calculate_multi_tier_profit(probs, t1, t2)

            if profit > best_profit:
                best_profit = profit
                best_t1 = t1
                best_t2 = t2

    print("===== BEST MULTI-TIER STRATEGY =====")
    print(f"Tier 1 threshold: {best_t1:.4f}")
    print(f"Tier 2 threshold: {best_t2:.4f}")
    print(f"Maximum Expected Profit: {best_profit:.2f}")


if __name__ == "__main__":
    main()

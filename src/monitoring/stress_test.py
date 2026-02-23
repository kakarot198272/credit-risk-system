import pandas as pd
import numpy as np
import json

# Paths
PRED_PATH = "artifacts/metrics/validation_predictions_xgboost.csv"
CONFIG_PATH = "config/decision_policy.json"

# Stress assumption
PD_SHOCK = 0.02  # +2% absolute PD increase


def load_policy():
    with open(CONFIG_PATH, "r") as f:
        policy = json.load(f)
    return policy


def compute_profit(probs, margin, lgd, threshold):
    approve = probs < threshold
    approved_probs = probs[approve]

    if len(approved_probs) == 0:
        return 0.0

    expected_loss = np.sum(approved_probs * lgd)
    expected_revenue = np.sum((1 - approved_probs) * margin)
    expected_profit = expected_revenue - expected_loss

    return expected_profit


def main():
    # Load predictions
    df = pd.read_csv(PRED_PATH)
    base_probs = df["pred_prob"].values

    # Load decision policy
    policy = load_policy()
    margin = policy["margin"]
    lgd = policy["lgd"]
    threshold = policy["threshold"]

    # Baseline
    baseline_profit = compute_profit(base_probs, margin, lgd, threshold)

    # Apply stress (cap PD at 1.0)
    stressed_probs = np.minimum(base_probs + PD_SHOCK, 1.0)
    stressed_profit = compute_profit(stressed_probs, margin, lgd, threshold)

    print("===== STRESS TEST RESULTS =====")
    print(f"PD Shock Applied: +{PD_SHOCK:.2f}")
    print(f"Threshold Used: {threshold:.4f}")
    print(f"Baseline Profit: {baseline_profit:.2f}")
    print(f"Stressed Profit: {stressed_profit:.2f}")
    print(f"Profit Change: {stressed_profit - baseline_profit:.2f}")

    if baseline_profit != 0:
        print(
            f"Profit Change (%): "
            f"{100 * (stressed_profit - baseline_profit) / baseline_profit:.2f}%"
        )
    else:
        print("Baseline profit is zero; cannot compute percentage change.")


if __name__ == "__main__":
    main()

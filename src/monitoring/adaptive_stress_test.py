import pandas as pd
import numpy as np
import json

# Paths
PRED_PATH = "artifacts/metrics/validation_predictions_xgboost.csv"
CONFIG_PATH = "config/decision_policy.json"

# Stress
PD_SHOCK = 0.02

THRESHOLDS = np.linspace(0.01, 0.5, 200)


def load_policy():
    with open(CONFIG_PATH, "r") as f:
        policy = json.load(f)
    return policy


def compute_profit(probs, threshold, margin, lgd):
    approve = probs < threshold
    approved_probs = probs[approve]

    expected_loss = np.sum(approved_probs * lgd)
    expected_revenue = np.sum((1 - approved_probs) * margin)
    expected_profit = expected_revenue - expected_loss

    return expected_profit


def optimize_threshold(probs, margin, lgd):
    best_profit = -np.inf
    best_t = 0

    for t in THRESHOLDS:
        profit = compute_profit(probs, t, margin, lgd)
        if profit > best_profit:
            best_profit = profit
            best_t = t

    return best_t, best_profit


def main():
    # Load data
    df = pd.read_csv(PRED_PATH)
    base_probs = df["pred_prob"].values

    # Load policy
    policy = load_policy()
    margin = policy["margin"]
    lgd = policy["lgd"]
    base_threshold = policy["threshold"]

    # Baseline profit
    baseline_profit = compute_profit(base_probs, base_threshold, margin, lgd)

    # Apply stress
    stressed_probs = np.minimum(base_probs + PD_SHOCK, 1.0)

    # Stressed profit (fixed threshold)
    stressed_fixed_profit = compute_profit(
        stressed_probs, base_threshold, margin, lgd
    )

    # Stressed profit (adaptive threshold)
    new_threshold, stressed_adaptive_profit = optimize_threshold(
        stressed_probs, margin, lgd
    )

    print("===== ADAPTIVE STRESS TEST =====")
    print(f"PD Shock Applied: +{PD_SHOCK:.2f}")

    print("\n--- Baseline ---")
    print(f"Threshold: {base_threshold:.4f}")
    print(f"Profit: {baseline_profit:.2f}")

    print("\n--- Stressed (Fixed Threshold) ---")
    print(f"Threshold: {base_threshold:.4f}")
    print(f"Profit: {stressed_fixed_profit:.2f}")

    print("\n--- Stressed (Adaptive Threshold) ---")
    print(f"New Optimal Threshold: {new_threshold:.4f}")
    print(f"Profit: {stressed_adaptive_profit:.2f}")

    print("\n--- Recovery ---")
    recovery = stressed_adaptive_profit - stressed_fixed_profit
    lost_profit = baseline_profit - stressed_fixed_profit

    print(f"Profit Recovered: {recovery:.2f}")
    if lost_profit > 0:
        print(f"Recovery (% of lost profit): {100 * recovery / lost_profit:.2f}%")
    else:
        print("No profit loss to recover.")


if __name__ == "__main__":
    main()

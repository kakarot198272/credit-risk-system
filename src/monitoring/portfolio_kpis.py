import pandas as pd
import numpy as np
import json

# Paths
PRED_PATH = "artifacts/metrics/validation_predictions_xgboost.csv"
CONFIG_PATH = "config/decision_policy.json"


def load_policy():
    with open(CONFIG_PATH, "r") as f:
        policy = json.load(f)
    return policy


def main():
    # Load predictions
    df = pd.read_csv(PRED_PATH)
    probs = df["pred_prob"].values

    # Load decision policy
    policy = load_policy()
    margin = policy["margin"]
    lgd = policy["lgd"]
    threshold = policy["threshold"]

    # Decision: approve if probability < threshold
    approve = probs < threshold

    total_apps = len(probs)
    approved_count = approve.sum()
    rejected_count = total_apps - approved_count

    approval_rate = approved_count / total_apps

    # Risk metrics
    approved_probs = probs[approve]

    if len(approved_probs) > 0:
        expected_default_rate = approved_probs.mean()
        expected_loss = np.sum(approved_probs * lgd)
        expected_revenue = np.sum((1 - approved_probs) * margin)
        expected_profit = expected_revenue - expected_loss
    else:
        expected_default_rate = 0.0
        expected_loss = 0.0
        expected_revenue = 0.0
        expected_profit = 0.0

    print("===== PORTFOLIO MONITORING KPIs =====")
    print(f"Total Applications: {total_apps}")
    print(f"Approved: {approved_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Approval Rate: {approval_rate:.4f}")

    print("\n--- Risk Metrics ---")
    print(f"Threshold Used: {threshold:.4f}")
    print(f"Expected Default Rate (Approved): {expected_default_rate:.4f}")
    print(f"Expected Loss (EL): {expected_loss:.2f}")

    print("\n--- Revenue Metrics ---")
    print(f"Expected Revenue: {expected_revenue:.2f}")
    print(f"Expected Profit: {expected_profit:.2f}")


if __name__ == "__main__":
    main()

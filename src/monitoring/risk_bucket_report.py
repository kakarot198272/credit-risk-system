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


def assign_bucket(p):
    if p < 0.05:
        return "0-5%"
    elif p < 0.10:
        return "5-10%"
    elif p < 0.15:
        return "10-15%"
    elif p < 0.20:
        return "15-20%"
    else:
        return "20%+"


def main():
    df = pd.read_csv(PRED_PATH)

    # Load decision policy
    policy = load_policy()
    margin = policy["margin"]
    lgd = policy["lgd"]
    threshold = policy["threshold"]

    # Apply decision rule
    df["approved"] = df["pred_prob"] < threshold

    # Only analyze approved portfolio
    approved_df = df[df["approved"]].copy()

    if len(approved_df) == 0:
        print("No approved loans under current threshold.")
        return

    # Assign risk buckets
    approved_df["risk_bucket"] = approved_df["pred_prob"].apply(assign_bucket)

    # Compute expected metrics
    approved_df["expected_loss"] = approved_df["pred_prob"] * lgd
    approved_df["expected_revenue"] = (1 - approved_df["pred_prob"]) * margin
    approved_df["expected_profit"] = (
        approved_df["expected_revenue"] - approved_df["expected_loss"]
    )

    # Group by bucket
    summary = approved_df.groupby("risk_bucket").agg(
        count=("pred_prob", "count"),
        avg_pd=("pred_prob", "mean"),
        total_loss=("expected_loss", "sum"),
        total_revenue=("expected_revenue", "sum"),
        total_profit=("expected_profit", "sum"),
    ).reset_index()

    # Sort buckets logically
    bucket_order = ["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]
    summary["risk_bucket"] = pd.Categorical(
        summary["risk_bucket"],
        categories=bucket_order,
        ordered=True
    )
    summary = summary.sort_values("risk_bucket")

    print("\n===== RISK BUCKET REPORT (Approved Portfolio) =====\n")
    print(summary.to_string(index=False))

    print("\n===== PORTFOLIO TOTALS =====")
    print(f"Threshold Used: {threshold:.4f}")
    print(f"Total Approved: {len(approved_df)}")
    print(f"Total Expected Loss: {approved_df['expected_loss'].sum():.2f}")
    print(f"Total Expected Revenue: {approved_df['expected_revenue'].sum():.2f}")
    print(f"Total Expected Profit: {approved_df['expected_profit'].sum():.2f}")


if __name__ == "__main__":
    main()

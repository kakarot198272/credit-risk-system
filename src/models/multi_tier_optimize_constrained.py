import numpy as np
import pandas as pd

# Use your validation predictions from the chosen model (XGBoost)
PRED_PATH = "artifacts/metrics/validation_predictions_xgboost.csv"

# Business assumptions (constrained / realistic)
LGD = 0.60

TIER1_MARGIN = 0.10          # Base product margin (fixed)
TIER2_MARGIN = 0.14          # Capped uplift margin (realistic)
TIER2_TAKE_RATE = 0.75       # Only 75% accept higher APR offer (demand drops)

# Search settings
THRESHOLDS = np.linspace(0.01, 0.50, 50)


def expected_profit_per_loan(p, margin):
    """
    Expected profit for ONE offered loan:
    (1 - p)*margin - p*LGD
    """
    return (1 - p) * margin - p * LGD


def single_tier_profit(probs):
    """
    Baseline: approve if p < t using margin=0.10, else reject.
    Find t that maximizes expected profit.
    """
    best_profit = -np.inf
    best_t = 0.0
    best_approve_rate = 0.0

    for t in THRESHOLDS:
        approve = probs < t
        profits = np.zeros_like(probs)

        profits[approve] = expected_profit_per_loan(probs[approve], TIER1_MARGIN)

        total_profit = profits.sum()
        if total_profit > best_profit:
            best_profit = total_profit
            best_t = t
            best_approve_rate = approve.mean()

    return best_t, best_profit, best_approve_rate


def multi_tier_profit(probs, t1, t2):
    """
    Tier 1: p < t1, margin=0.10, take_rate=1.0
    Tier 2: t1 <= p < t2, margin=0.14, take_rate=0.75
    Reject: p >= t2, profit=0
    """
    profits = np.zeros_like(probs)

    tier1 = probs < t1
    tier2 = (probs >= t1) & (probs < t2)

    # Tier 1 profits (everyone accepts)
    profits[tier1] = expected_profit_per_loan(probs[tier1], TIER1_MARGIN)

    # Tier 2 profits (only some accept)
    profits[tier2] = TIER2_TAKE_RATE * expected_profit_per_loan(probs[tier2], TIER2_MARGIN)

    total_profit = profits.sum()
    return total_profit, tier1.mean(), tier2.mean()


def main():
    df = pd.read_csv(PRED_PATH)
    probs = df["pred_prob"].values

    # Baseline: single-tier (fixed pricing)
    base_t, base_profit, base_approve_rate = single_tier_profit(probs)

    # Constrained multi-tier optimization
    best_profit = -np.inf
    best_t1 = 0.0
    best_t2 = 0.0
    best_t1_rate = 0.0
    best_t2_rate = 0.0

    for t1 in THRESHOLDS:
        for t2 in THRESHOLDS:
            if t2 <= t1:
                continue

            profit, t1_rate, t2_rate = multi_tier_profit(probs, t1, t2)

            if profit > best_profit:
                best_profit = profit
                best_t1 = t1
                best_t2 = t2
                best_t1_rate = t1_rate
                best_t2_rate = t2_rate

    print("===== BASELINE (Single-Tier / Fixed Pricing) =====")
    print(f"Margin: {TIER1_MARGIN:.2f}, LGD: {LGD:.2f}")
    print(f"Optimal Threshold: {base_t:.4f}")
    print(f"Approve Rate: {base_approve_rate:.4f}")
    print(f"Maximum Expected Profit: {base_profit:.2f}")

    print("\n===== CONSTRAINED MULTI-TIER STRATEGY =====")
    print(f"Tier1 Margin: {TIER1_MARGIN:.2f}")
    print(f"Tier2 Margin (cap): {TIER2_MARGIN:.2f}")
    print(f"Tier2 Take Rate: {TIER2_TAKE_RATE:.2f}")
    print(f"Tier1 Threshold (t1): {best_t1:.4f}")
    print(f"Tier2 Threshold (t2): {best_t2:.4f}")
    print(f"Tier1 Rate: {best_t1_rate:.4f}")
    print(f"Tier2 Rate: {best_t2_rate:.4f}")
    print(f"Reject Rate: {1 - (best_t1_rate + best_t2_rate):.4f}")
    print(f"Maximum Expected Profit: {best_profit:.2f}")

    print("\n===== LIFT vs BASELINE =====")
    print(f"Profit Lift: {(best_profit - base_profit):.2f}")
    if base_profit != 0:
        print(f"Profit Lift (%): {100 * (best_profit - base_profit) / base_profit:.2f}%")
    else:
        print("Profit Lift (%): N/A (baseline profit is 0)")


if __name__ == "__main__":
    main()

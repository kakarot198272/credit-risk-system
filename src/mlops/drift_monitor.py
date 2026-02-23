import pandas as pd
import numpy as np
import os

BASELINE_PATH = "artifacts/metrics/validation_predictions_xgboost.csv"
LOG_PATH = "logs/prediction_logs.jsonl"

N_BUCKETS = 10
PSI_THRESHOLD = 0.2


def calculate_psi(expected, actual, buckets=10):

    # Use quantile bins based on baseline distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    expected_percents = []
    actual_percents = []

    for i in range(buckets):
        lower = breakpoints[i]
        upper = breakpoints[i + 1]

        expected_count = ((expected >= lower) & (expected < upper)).sum()
        actual_count = ((actual >= lower) & (actual < upper)).sum()

        expected_percent = expected_count / len(expected)
        actual_percent = actual_count / len(actual)

        if expected_percent == 0:
            expected_percent = 1e-6
        if actual_percent == 0:
            actual_percent = 1e-6

        expected_percents.append(expected_percent)
        actual_percents.append(actual_percent)

    psi = np.sum(
        (np.array(actual_percents) - np.array(expected_percents)) *
        np.log(np.array(actual_percents) / np.array(expected_percents))
    )

    return psi


def main():

    if not os.path.exists(LOG_PATH):
        print("No live prediction logs found.")
        return

    baseline_df = pd.read_csv(BASELINE_PATH)
    baseline_pd = baseline_df["pred_prob"].values

    live_df = pd.read_json(LOG_PATH, lines=True)
    live_pd = live_df["predicted_pd"].values

    if len(live_pd) < 10:
        print("Not enough live data for drift analysis.")
        return

    psi = calculate_psi(baseline_pd, live_pd, N_BUCKETS)

    print("\n===== DRIFT MONITOR =====")
    print(f"Baseline Samples: {len(baseline_pd)}")
    print(f"Live Samples: {len(live_pd)}")
    print(f"PSI: {psi:.4f}")

    if psi > PSI_THRESHOLD:
        print("⚠ ALERT: Significant drift detected!")
    else:
        print("✅ Model stable. No significant drift detected.")


if __name__ == "__main__":
    main()

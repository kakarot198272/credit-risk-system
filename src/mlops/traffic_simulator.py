import pandas as pd
import requests
import random
import time

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "artifacts/metrics/validation_predictions_xgboost.csv"

N_REQUESTS = 50
SLEEP_SECONDS = 0.1


def main():

    print("Loading validation dataset...")
    df = pd.read_csv(DATA_PATH)

    # Remove prediction column if present
    if "pred_prob" in df.columns:
        df = df.drop(columns=["pred_prob"])

    print(f"Simulating {N_REQUESTS} live requests...\n")

    for i in range(N_REQUESTS):
        row = df.sample(1).iloc[0].to_dict()

        try:
            response = requests.post(API_URL, json=row)
            print(f"Request {i+1} → Status: {response.status_code}")
        except Exception as e:
            print(f"Request failed: {e}")

        time.sleep(SLEEP_SECONDS)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()

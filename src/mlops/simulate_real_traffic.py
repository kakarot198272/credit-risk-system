import requests
import pandas as pd
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "data/features/model_input.parquet"

def clean_payload(row_dict):
    """
    Convert NaN values to None for JSON compatibility.
    """
    for k, v in row_dict.items():
        if isinstance(v, float) and np.isnan(v):
            row_dict[k] = None
    return row_dict


def main():
    df = pd.read_parquet(DATA_PATH)

    # Drop target column
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])

    # Sample 100 realistic customers
    sample_df = df.sample(100, random_state=42)

    print("Sending 100 realistic samples to API...")

    for _, row in sample_df.iterrows():
        payload = clean_payload(row.to_dict())
        response = requests.post(API_URL, json=payload)

    print("Simulation complete.")


if __name__ == "__main__":
    main()

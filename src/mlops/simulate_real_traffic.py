"""Send rows from a supplied dataset to the API; preserves the API's input schema."""

import argparse

import pandas as pd
import requests

from src.models.data import read_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    schema_response = requests.get(args.api_url + "/schema", timeout=10)
    schema_response.raise_for_status()
    names = list(schema_response.json()["input_features"])
    data = read_dataset(args.data)
    sample = data.sample(min(args.count, len(data)), random_state=42)[names]
    for _, row in sample.iterrows():
        payload = {name: None if pd.isna(value) else value for name, value in row.items()}
        response = requests.post(
            args.api_url + "/predict", json={"features": payload, "explain": False}, timeout=30
        )
        response.raise_for_status()
    print(
        f"Sent {len(sample)} successful requests; these are replayed dataset rows, not production users."
    )


if __name__ == "__main__":
    main()

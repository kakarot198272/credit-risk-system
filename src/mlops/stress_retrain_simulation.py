import pandas as pd
import numpy as np
import os

FEATURE_PATH = "data/features/model_input.parquet"
STRESS_PATH = "data/features/model_input_stressed.parquet"


def create_high_risk_data(df, multiplier=1.0):
    """
    Create synthetic high-risk borrowers.
    """

    stressed = df.sample(int(len(df) * multiplier), replace=True, random_state=42).copy()

    # Force target to default (high-risk)
    stressed["TARGET"] = 1

    # Increase risk features artificially
    if "credit_to_income_ratio" in stressed.columns:
        stressed["credit_to_income_ratio"] *= 2

    if "annuity_to_income_ratio" in stressed.columns:
        stressed["annuity_to_income_ratio"] *= 2

    return stressed


def main():
    df = pd.read_parquet(FEATURE_PATH)

    print("Original dataset size:", len(df))

    stressed_data = create_high_risk_data(df, multiplier=1.0)

    combined = pd.concat([df, stressed_data], ignore_index=True)

    print("New stressed dataset size:", len(combined))

    combined.to_parquet(STRESS_PATH)

    print("Stressed dataset saved.")


if __name__ == "__main__":
    main()

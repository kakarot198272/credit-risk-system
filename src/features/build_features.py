import os
import pandas as pd

PROCESSED_PATH = "data/processed"
RAW_PATH = "data/raw"
FEATURE_PATH = "data/features"


# =============================
# Bureau Aggregation
# =============================
def aggregate_bureau():
    bureau_path = os.path.join(RAW_PATH, "bureau.csv")
    bureau = pd.read_csv(bureau_path)

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_total_loans=("SK_ID_BUREAU", "count"),
        bureau_active_loans=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_avg_credit_amt=("AMT_CREDIT_SUM", "mean"),
        bureau_max_credit_amt=("AMT_CREDIT_SUM", "max"),
    ).reset_index()

    return bureau_agg


# =============================
# Previous Application Aggregation
# =============================
def aggregate_previous():
    prev_path = os.path.join(RAW_PATH, "previous_application.csv")
    prev = pd.read_csv(prev_path)

    prev_agg = prev.groupby("SK_ID_CURR").agg(
        prev_app_count=("SK_ID_PREV", "count"),
        prev_avg_credit=("AMT_CREDIT", "mean"),
        prev_approved_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum())
    ).reset_index()

    return prev_agg


# =============================
# Build Final Feature Dataset
# =============================
def build_features():
    os.makedirs(FEATURE_PATH, exist_ok=True)

    # Load processed application data
    app_path = os.path.join(PROCESSED_PATH, "application_train.parquet")
    application = pd.read_parquet(app_path)

    # Aggregate external tables
    bureau_features = aggregate_bureau()
    prev_features = aggregate_previous()

    # Merge bureau features
    final_df = application.merge(
        bureau_features,
        on="SK_ID_CURR",
        how="left"
    )

    # Merge previous application features
    final_df = final_df.merge(
        prev_features,
        on="SK_ID_CURR",
        how="left"
    )

    # =============================
    # Fill Missing Aggregation Values
    # =============================
    agg_cols = [
        "bureau_total_loans",
        "bureau_active_loans",
        "bureau_avg_credit_amt",
        "bureau_max_credit_amt",
        "prev_app_count",
        "prev_avg_credit",
        "prev_approved_count"
    ]

    for col in agg_cols:
        final_df[col].fillna(0, inplace=True)

    # =============================
    # Financial Leverage Ratios
    # =============================
    final_df["credit_to_income_ratio"] = (
        final_df["AMT_CREDIT"] / final_df["AMT_INCOME_TOTAL"]
    )

    final_df["annuity_to_income_ratio"] = (
        final_df["AMT_ANNUITY"] / final_df["AMT_INCOME_TOTAL"]
    )

    final_df["credit_to_annuity_ratio"] = (
        final_df["AMT_CREDIT"] / final_df["AMT_ANNUITY"]
    )

    # Clean infinities
    ratio_cols = [
        "credit_to_income_ratio",
        "annuity_to_income_ratio",
        "credit_to_annuity_ratio"
    ]

    for col in ratio_cols:
        final_df[col].replace([float("inf"), -float("inf")], 0, inplace=True)
        final_df[col].fillna(0, inplace=True)

    # =============================
    # Bureau Ratios
    # =============================
    final_df["bureau_active_ratio"] = (
        final_df["bureau_active_loans"] / final_df["bureau_total_loans"]
    )
    final_df["bureau_active_ratio"].fillna(0, inplace=True)

    # =============================
    # Previous Application Ratios
    # =============================
    final_df["prev_approval_rate"] = (
        final_df["prev_approved_count"] / final_df["prev_app_count"]
    )
    final_df["prev_approval_rate"].fillna(0, inplace=True)

    # Save final dataset
    output_file = os.path.join(FEATURE_PATH, "model_input.parquet")
    final_df.to_parquet(output_file, index=False)

    print("Feature engineering completed.")
    print(f"Saved model dataset to {output_file}")


if __name__ == "__main__":
    build_features()

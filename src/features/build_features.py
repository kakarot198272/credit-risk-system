"""Build one row per applicant from Home Credit-shaped source CSV files.

Only deterministic row/ID aggregations occur here. Learned imputers and encoders
are fit later on training applicants only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.artifacts import ROOT, write_json

DERIVED_DEPENDENCIES = {
    "credit_to_income_ratio": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "annuity_to_income_ratio": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "credit_to_annuity_ratio": ("AMT_CREDIT", "AMT_ANNUITY"),
    "bureau_active_ratio": ("bureau_active_loans", "bureau_total_loans"),
    "prev_approval_rate": ("prev_approved_count", "prev_app_count"),
}


def engineer_financial_ratios(frame):
    result = frame.copy()
    for name, (numerator, denominator) in DERIVED_DEPENDENCIES.items():
        if numerator in result and denominator in result:
            top = pd.to_numeric(result[numerator], errors="raise")
            bottom = pd.to_numeric(result[denominator], errors="raise")
            # Undefined ratios remain missing, rather than looking like zero risk.
            result[name] = top.div(bottom.where(bottom > 0)).replace([np.inf, -np.inf], np.nan)
        elif name in result:
            raise ValueError(f"{name} needs source columns {numerator} and {denominator}")
    return result


def build_features(raw_dir=ROOT / "data/raw", output=ROOT / "data/features/model_input.parquet"):
    raw_dir, output = Path(raw_dir), Path(output)
    required = ["application_train.csv", "bureau.csv", "previous_application.csv"]
    if missing := [name for name in required if not (raw_dir / name).is_file()]:
        raise FileNotFoundError(f"Place the original source files in {raw_dir}: missing {missing}")
    application = pd.read_csv(raw_dir / "application_train.csv")
    if application.SK_ID_CURR.duplicated().any():
        raise ValueError("application_train.csv must have one row per applicant")
    bureau = pd.read_csv(
        raw_dir / "bureau.csv",
        usecols=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM"],
    )
    previous = pd.read_csv(
        raw_dir / "previous_application.csv",
        usecols=["SK_ID_CURR", "SK_ID_PREV", "AMT_CREDIT", "NAME_CONTRACT_STATUS"],
    )
    bureau_features = bureau.groupby("SK_ID_CURR").agg(
        bureau_total_loans=("SK_ID_BUREAU", "count"),
        bureau_active_loans=("CREDIT_ACTIVE", lambda x: x.eq("Active").sum()),
        bureau_avg_credit_amt=("AMT_CREDIT_SUM", "mean"),
        bureau_max_credit_amt=("AMT_CREDIT_SUM", "max"),
    )
    previous_features = previous.groupby("SK_ID_CURR").agg(
        prev_app_count=("SK_ID_PREV", "count"),
        prev_avg_credit=("AMT_CREDIT", "mean"),
        prev_approved_count=("NAME_CONTRACT_STATUS", lambda x: x.eq("Approved").sum()),
    )
    result = application.merge(bureau_features, on="SK_ID_CURR", how="left", validate="one_to_one")
    result = result.merge(previous_features, on="SK_ID_CURR", how="left", validate="one_to_one")
    counts = ["bureau_total_loans", "bureau_active_loans", "prev_app_count", "prev_approved_count"]
    result[counts] = result[counts].fillna(0)
    if "DAYS_EMPLOYED" in result:
        result["DAYS_EMPLOYED"] = result.DAYS_EMPLOYED.replace(365243, np.nan)
    before = len(result)
    amount = pd.to_numeric(result.AMT_CREDIT, errors="coerce")
    result = result.loc[amount.notna() & np.isfinite(amount) & amount.gt(0)].copy()
    for name in ["AMT_INCOME_TOTAL", "AMT_ANNUITY"]:
        numeric = pd.to_numeric(result[name], errors="coerce")
        result[name] = numeric.where(np.isfinite(numeric) & numeric.gt(0))
    result = engineer_financial_ratios(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(
        output.with_suffix(".metadata.json"),
        {
            "source": "User-supplied Home Credit-shaped original source files",
            "dataset_kind": "real",
            "input_applicants": before,
            "output_applicants": len(result),
            "excluded_invalid_loan_amount": before - len(result),
            "note": "Confirm all supplied source fields are available at application time. No learned preprocessing is fitted here.",
        },
    )
    print(
        f"Saved {len(result):,} applicants to {output}; excluded {before - len(result)} invalid loan amounts."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/features/model_input.parquet")
    args = parser.parse_args()
    build_features(args.raw_dir, args.output)


if __name__ == "__main__":
    main()

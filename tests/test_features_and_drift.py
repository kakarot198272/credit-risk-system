import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features, engineer_financial_ratios
from src.mlops.drift_monitor import calculate_psi, drift_report


def test_zero_denominators_stay_missing():
    frame = pd.DataFrame({"AMT_CREDIT": [100], "AMT_INCOME_TOTAL": [0], "AMT_ANNUITY": [0]})
    result = engineer_financial_ratios(frame)
    assert (
        result[["credit_to_income_ratio", "annuity_to_income_ratio", "credit_to_annuity_ratio"]]
        .isna()
        .all()
        .all()
    )


def test_raw_tables_aggregate_without_multiplying_applicants(tmp_path):
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "AMT_CREDIT": [100, 200],
            "AMT_INCOME_TOTAL": [200, 200],
            "AMT_ANNUITY": [20, 40],
        }
    ).to_csv(tmp_path / "application_train.csv", index=False)
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1],
            "SK_ID_BUREAU": [10, 11],
            "CREDIT_ACTIVE": ["Active", "Closed"],
            "AMT_CREDIT_SUM": [100, 300],
        }
    ).to_csv(tmp_path / "bureau.csv", index=False)
    pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1],
            "SK_ID_PREV": [20, 21],
            "AMT_CREDIT": [50, 150],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused"],
        }
    ).to_csv(tmp_path / "previous_application.csv", index=False)
    result = build_features(tmp_path, tmp_path / "features.parquet")
    assert len(result) == 2
    assert result.loc[0, "bureau_avg_credit_amt"] == 200
    assert result.loc[0, "prev_approval_rate"] == 0.5
    assert result.loc[1, "bureau_total_loans"] == 0
    assert pd.isna(result.loc[1, "bureau_avg_credit_amt"])


def test_psi_catches_out_of_range_and_constant_distributions():
    baseline = np.linspace(0.05, 0.20, 500)
    assert calculate_psi(baseline, baseline) == pytest.approx(0)
    assert calculate_psi(baseline, np.full(500, 0.9)) > 0.2
    assert np.isfinite(calculate_psi(np.full(500, 0.1), np.full(500, 0.9)))


def test_drift_ignores_other_model_versions(trained_run, tmp_path):
    logs = pd.DataFrame(
        {
            "model_version": ["other-version"] * 30,
            "dataset_kind": ["synthetic"] * 30,
            "predicted_pd": [0.9] * 30,
        }
    )
    path = tmp_path / "logs.jsonl"
    logs.to_json(path, lines=True, orient="records")
    result = drift_report(trained_run["run"], path)
    assert result["status"] == "insufficient_data"
    assert result["live_samples"] == 0

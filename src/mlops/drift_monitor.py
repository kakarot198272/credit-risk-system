"""Version-matched prediction-score drift. Not a measure of model accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk.artifacts import ROOT, active_bundle_path, model_directory


def calculate_psi(expected, actual, buckets=10):
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if expected.ndim != 1 or actual.ndim != 1 or not len(expected) or not len(actual):
        raise ValueError("PSI needs two nonempty one-dimensional samples")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("PSI inputs must be finite")
    if not isinstance(buckets, int) or buckets < 2:
        raise ValueError("Need at least two bins")
    interior = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)[1:-1]))
    edges = np.r_[-np.inf, interior, np.inf]
    baseline_counts = np.histogram(expected, bins=edges)[0]
    live_counts = np.histogram(actual, bins=edges)[0]
    # Smoothing and normalization retain all probability mass, including extremes.
    baseline = (baseline_counts + 0.5) / (len(expected) + 0.5 * len(baseline_counts))
    live = (live_counts + 0.5) / (len(actual) + 0.5 * len(live_counts))
    return float(np.sum((live - baseline) * np.log(live / baseline)))


def drift_report(run_dir, log_path):
    run_dir, log_path = Path(run_dir), Path(log_path)
    report = json.loads((run_dir / "report.json").read_text())
    baseline = pd.read_csv(run_dir / "validation_predictions.csv")["pred_prob"].to_numpy()
    if not log_path.exists() or log_path.stat().st_size == 0:
        return {
            "status": "insufficient_data",
            "live_samples": 0,
            "model_version": report["model_version"],
        }
    logs = pd.read_json(log_path, lines=True)
    required = {"model_version", "dataset_kind", "predicted_pd"}
    if not required.issubset(logs):
        raise ValueError("Prediction logs do not follow the version-2 schema")
    selected = logs.loc[
        (logs.model_version == report["model_version"])
        & (logs.dataset_kind == report["dataset_kind"])
    ]
    if len(selected) < 20:
        return {
            "status": "insufficient_data",
            "live_samples": len(selected),
            "model_version": report["model_version"],
        }
    score = calculate_psi(baseline, selected.predicted_pd.to_numpy())
    return {
        "status": "review_shift" if score >= 0.2 else "no_score_shift_flag",
        "psi": score,
        "live_samples": len(selected),
        "baseline_samples": len(baseline),
        "excluded_other_versions": len(logs) - len(selected),
        "model_version": report["model_version"],
        "note": "0.2 is a demo heuristic. Score drift alone does not establish accuracy loss, concept drift, or a need to retrain.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=model_directory())
    parser.add_argument("--log-path", type=Path, default=ROOT / "logs/prediction_logs.jsonl")
    args = parser.parse_args()
    print(
        json.dumps(drift_report(active_bundle_path(args.model_dir).parent, args.log_path), indent=2)
    )


if __name__ == "__main__":
    main()

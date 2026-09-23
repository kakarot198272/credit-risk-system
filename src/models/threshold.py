"""Explore validation policies; never overwrite the frozen active policy."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.risk.artifacts import active_bundle_path, model_directory
from src.risk.policy import optimize_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=model_directory())
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--lgd", type=float, default=0.60)
    args = parser.parse_args()
    predictions = pd.read_csv(
        active_bundle_path(args.model_dir).parent / "validation_predictions.csv"
    )
    policy, summary, _ = optimize_policy(
        predictions.true_label,
        predictions.pred_prob,
        predictions.loan_amount,
        margin=args.margin,
        lgd=args.lgd,
    )
    print(
        json.dumps(
            {"split": "validation", "policy": policy.to_dict(), "portfolio": summary}, indent=2
        )
    )


if __name__ == "__main__":
    main()

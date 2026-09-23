"""Hypothetical +2 percentage point PD shock under a frozen policy."""

import json

import pandas as pd

from src.risk.artifacts import active_bundle_path
from src.risk.policy import DecisionPolicy, stress_summary


def main():
    run = active_bundle_path().parent
    report = json.loads((run / "report.json").read_text())
    frame = pd.read_csv(run / "validation_predictions.csv")
    print(
        json.dumps(
            stress_summary(frame.pred_prob, frame.loan_amount, DecisionPolicy(**report["policy"])),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

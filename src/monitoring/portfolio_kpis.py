"""Validation portfolio report using the versioned active policy."""

import json

import pandas as pd

from src.risk.artifacts import active_bundle_path
from src.risk.policy import DecisionPolicy, portfolio_summary


def main():
    run = active_bundle_path().parent
    report = json.loads((run / "report.json").read_text())
    frame = pd.read_csv(run / "validation_predictions.csv")
    print(
        json.dumps(
            portfolio_summary(
                frame.true_label,
                frame.pred_prob,
                frame.loan_amount,
                DecisionPolicy(**report["policy"]),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

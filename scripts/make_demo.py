"""Generate explicitly synthetic applicants for software demonstrations only."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import engineer_financial_ratios
from src.risk.artifacts import ROOT, write_json


def generate_demo(rows=4000, seed=2026):
    if rows < 500:
        raise ValueError("Use at least 500 demo applicants")
    rng = np.random.default_rng(seed)
    income = rng.lognormal(np.log(65000), 0.50, rows)
    ratio = rng.uniform(0.3, 5, rows)
    credit = income * ratio
    active = rng.poisson(1.5, rows)
    total = active + rng.poisson(2, rows)
    previous = rng.poisson(3, rows)
    approved = rng.binomial(previous, 0.70)
    annuity = credit / rng.uniform(3, 9, rows)
    employment = rng.choice(
        ["Working", "Commercial associate", "Pensioner"], rows, p=[0.6, 0.25, 0.15]
    )
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100001, 100001 + rows),
            "AMT_INCOME_TOTAL": income.round(2),
            "AMT_CREDIT": credit.round(2),
            "AMT_ANNUITY": annuity.round(2),
            "NAME_INCOME_TYPE": employment,
            "bureau_total_loans": total,
            "bureau_active_loans": active,
            "bureau_avg_credit_amt": rng.lognormal(np.log(25000), 0.6, rows).round(2),
            "bureau_max_credit_amt": rng.lognormal(np.log(70000), 0.5, rows).round(2),
            "prev_app_count": previous,
            "prev_approved_count": approved,
            "prev_avg_credit": rng.lognormal(np.log(45000), 0.6, rows).round(2),
        }
    )
    # A stochastic toy mechanism with non-linear interactions; labels are sampled,
    # not forced to default by a relabeling step and never supplied as features.
    log_odds = (
        -3.2
        + 0.5 * (ratio - 2)
        + 0.22 * active
        + 1.1 * ((ratio > 3.5) & (active >= 2))
        + 0.4 * (annuity / income > 0.6)
    )
    probabilities = 1 / (1 + np.exp(-log_odds))
    frame["TARGET"] = rng.binomial(1, probabilities)
    frame["__synthetic__"] = True
    missing = rng.random(rows) < 0.05
    frame.loc[missing, "bureau_avg_credit_amt"] = np.nan
    return engineer_financial_ratios(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=ROOT / "data/demo/model_input.parquet")
    args = parser.parse_args()
    frame = generate_demo(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    write_json(
        args.output.with_suffix(".metadata.json"),
        {
            "dataset_kind": "synthetic",
            "rows": len(frame),
            "seed": args.seed,
            "purpose": "Software demonstration only; not real loan performance",
        },
    )
    print(f"Created {len(frame):,} SYNTHETIC applicants: {args.output}")


if __name__ == "__main__":
    main()

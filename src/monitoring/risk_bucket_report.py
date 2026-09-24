"""Validation risk buckets: predicted risk versus observed default rate."""

import pandas as pd

from src.risk.artifacts import active_bundle_path
from src.risk.metrics import calibration_bins


def main():
    frame = pd.read_csv(active_bundle_path().parent / "validation_predictions.csv")
    print(pd.DataFrame(calibration_bins(frame.true_label, frame.pred_prob)).to_string(index=False))


if __name__ == "__main__":
    main()

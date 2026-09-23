"""Discrimination, probability quality, and diagnostic calibration bins."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def classification_metrics(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "default_prevalence": float(y.mean()),
    }


def calibration_bins(y, probabilities, bins=10):
    frame = pd.DataFrame({"outcome": y, "probability": probabilities})
    frame["bin"] = pd.cut(frame.probability, bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    table = (
        frame.groupby("bin", observed=True)
        .agg(
            count=("outcome", "size"),
            mean_predicted_pd=("probability", "mean"),
            observed_default_rate=("outcome", "mean"),
        )
        .reset_index(drop=True)
    )
    return table.to_dict(orient="records")


def cohort_report(y, probabilities, amounts, threshold):
    frame = pd.DataFrame({"outcome": y, "probability": probabilities, "amount": amounts})
    frame["approved"] = frame.probability < threshold
    frame["loan_size_group"] = pd.qcut(frame.amount, q=4, duplicates="drop").astype(str)
    return (
        frame.groupby("loan_size_group", observed=True)
        .agg(
            applicants=("outcome", "size"),
            approval_rate=("approved", "mean"),
            observed_default_rate=("outcome", "mean"),
            mean_predicted_pd=("probability", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

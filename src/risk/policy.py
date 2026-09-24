"""Loan-exposure-weighted simulation, never a claim of realized bank profit."""

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionPolicy:
    margin: float = 0.10
    lgd: float = 0.60
    threshold: float = 1 / 7

    def __post_init__(self):
        for name in ("margin", "lgd", "threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")

    def to_dict(self):
        return asdict(self)


def validate_vectors(y, probabilities, amounts):
    y = np.asarray(y, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if any(a.ndim != 1 for a in (y, probabilities, amounts)):
        raise ValueError("Expected one-dimensional vectors")
    if len(y) == 0 or len(y) != len(probabilities) or len(y) != len(amounts):
        raise ValueError("Inputs must be nonempty and have equal lengths")
    if not all(np.isfinite(a).all() for a in (y, probabilities, amounts)):
        raise ValueError("Inputs must be finite")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Default labels must be 0 or 1")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be between 0 and 1")
    if not (amounts > 0).all():
        raise ValueError("Loan amounts must be positive")
    return y, probabilities, amounts


def expected_profit_if_funded(probabilities, amounts, policy):
    probabilities = np.asarray(probabilities, dtype=float)
    return np.asarray(amounts) * ((1 - probabilities) * policy.margin - probabilities * policy.lgd)


def portfolio_summary(y, probabilities, amounts, policy, *, approve_all=False):
    y, p, amounts = validate_vectors(y, probabilities, amounts)
    approved = np.ones(len(y), dtype=bool) if approve_all else p < policy.threshold
    funded = amounts[approved]
    outcomes = y[approved]
    loss = float(np.sum(funded * outcomes * policy.lgd))
    revenue = float(np.sum(funded * (1 - outcomes) * policy.margin))
    expected = expected_profit_if_funded(p[approved], funded, policy)
    return {
        "applications": len(y),
        "approved": int(approved.sum()),
        "approval_rate": float(approved.mean()),
        "approved_exposure": float(funded.sum()),
        "observed_default_rate": float(outcomes.mean()) if len(outcomes) else None,
        "mean_predicted_pd": float(p[approved].mean()) if approved.any() else None,
        "simulated_loss": loss,
        "simulated_revenue": revenue,
        "simulated_profit": revenue - loss,
        "expected_profit": float(expected.sum()),
        "profit_per_application": (revenue - loss) / len(y),
        "assumptions": policy.to_dict(),
    }


def optimize_policy(y, probabilities, amounts, *, margin=0.10, lgd=0.60):
    """Select using validation outcomes only; ties favor the stricter policy."""
    y, p, amounts = validate_vectors(y, probabilities, amounts)
    break_even = margin / (margin + lgd) if margin + lgd else 0.0
    thresholds = np.unique(np.r_[np.linspace(0, 1, 201), break_even])
    results = []
    for threshold in thresholds:
        policy = DecisionPolicy(margin, lgd, float(threshold))
        result = portfolio_summary(y, p, amounts, policy)
        results.append({"threshold": float(threshold), **result})
    best = max(results, key=lambda r: (r["simulated_profit"], -r["threshold"]))
    return DecisionPolicy(margin, lgd, best["threshold"]), best, results


def stress_summary(probabilities, amounts, policy, pd_shock=0.02):
    """Hypothetical score shock. No fabricated observed default outcomes."""
    p = np.asarray(probabilities, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    validate_vectors(np.zeros(len(p)), p, amounts)
    if not math.isfinite(pd_shock) or not 0 <= pd_shock <= 1:
        raise ValueError("PD shock must be between 0 and 1")
    stressed = np.clip(p + pd_shock, 0, 1)
    existing = p < policy.threshold
    new_originations = stressed < policy.threshold
    return {
        "pd_shock": pd_shock,
        "baseline_expected_profit": float(
            expected_profit_if_funded(p[existing], amounts[existing], policy).sum()
        ),
        "existing_portfolio_stressed_expected_profit": float(
            expected_profit_if_funded(stressed[existing], amounts[existing], policy).sum()
        ),
        "new_originations_stressed_expected_profit": float(
            expected_profit_if_funded(
                stressed[new_originations], amounts[new_originations], policy
            ).sum()
        ),
        "baseline_approval_rate": float(existing.mean()),
        "stressed_new_approval_rate": float(new_originations.mean()),
        "interpretation": "Assumption-driven scenario, not an observed economic forecast.",
    }

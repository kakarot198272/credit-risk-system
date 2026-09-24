import numpy as np
import pytest

from src.risk.policy import DecisionPolicy, optimize_policy, portfolio_summary, stress_summary


def test_outcome_profit_uses_actual_labels_and_loan_exposure():
    policy = DecisionPolicy(threshold=0.2)
    result = portfolio_summary([0, 1], [0.05, 0.10], [100, 1000], policy)
    assert result["simulated_profit"] == pytest.approx(-590)
    assert result["expected_profit"] == pytest.approx(36.5)
    optimistic = portfolio_summary([0, 1], [0.001, 0.001], [100, 1000], policy)
    assert optimistic["simulated_profit"] == result["simulated_profit"]
    assert optimistic["expected_profit"] > result["expected_profit"]


def test_threshold_boundary_and_reject_all():
    result = portfolio_summary([0, 1], [0.05, 0.1], [100, 100], DecisionPolicy(threshold=0.1))
    assert result["approved"] == 1
    empty = portfolio_summary([0, 1], [0.05, 0.1], [100, 100], DecisionPolicy(threshold=0))
    assert empty["simulated_profit"] == 0
    assert empty["observed_default_rate"] is None


def test_optimizing_outcomes_can_reject_overoptimistic_model():
    policy, summary, _ = optimize_policy([1, 0], [0.01, 0.02], [100, 100])
    assert policy.threshold == 0
    assert summary["simulated_profit"] == 0


def test_stress_does_not_remove_already_funded_loans():
    result = stress_summary([0.10, 0.20], [100, 100], DecisionPolicy(), pd_shock=0.10)
    assert result["existing_portfolio_stressed_expected_profit"] == pytest.approx(-4)
    assert result["new_originations_stressed_expected_profit"] == 0


@pytest.mark.parametrize(
    "labels,probabilities,amounts",
    [
        ([0, 1], [0.1], [100, 100]),
        ([0, 1], [0.1, np.nan], [100, 100]),
        ([0, 2], [0.1, 0.2], [100, 100]),
        ([0, 1], [0.1, 0.2], [100, -1]),
        ([0, 1], [0.1, 1.1], [100, 100]),
    ],
)
def test_invalid_financial_inputs_fail(labels, probabilities, amounts):
    with pytest.raises(ValueError):
        portfolio_summary(labels, probabilities, amounts, DecisionPolicy())

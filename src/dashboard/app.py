"""Interactive validation-policy explorer and API-backed applicant scoring."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.inference.predict import Predictor
from src.mlops.drift_monitor import drift_report
from src.risk.artifacts import ROOT, active_bundle_path, file_sha256
from src.risk.policy import DecisionPolicy, portfolio_summary, stress_summary

st.set_page_config(page_title="Credit Risk Explorer", page_icon="◈", layout="wide")
ACCENT = "#087f72"
API_URL = os.environ.get("CREDIT_RISK_API_URL", "http://127.0.0.1:8000").rstrip("/")
LOG_PATH = Path(os.environ.get("CREDIT_RISK_LOG_PATH", ROOT / "logs/prediction_logs.jsonl"))


@st.cache_resource
def load_run(bundle_path, modification_time):
    predictor = Predictor(bundle_path)
    run = Path(bundle_path).parent
    manifest = json.loads((run / "manifest.json").read_text())
    for filename, checksum in manifest["sha256"].items():
        if file_sha256(run / filename) != checksum:
            raise ValueError(f"Artifact integrity check failed: {filename}")
    report = json.loads((run / "report.json").read_text())
    validation = pd.read_csv(run / "validation_predictions.csv")
    return predictor, report, validation


def currency(value):
    return f"{value:,.0f} CU"


def probability(value):
    return "No approvals" if value is None else f"{value:.1%}"


def calibration_figure(results):
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect agreement",
        line={"dash": "dash", "color": "#9ca3af"},
    )
    for name, result in results.items():
        bins = pd.DataFrame(result["calibration"])
        fig.add_scatter(
            x=bins.mean_predicted_pd,
            y=bins.observed_default_rate,
            mode="lines+markers",
            name=name.replace("_", " "),
        )
    fig.update_layout(
        xaxis_title="Mean predicted default probability",
        yaxis_title="Observed default rate",
        height=380,
        margin={"t": 25, "b": 30},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


st.title("Credit Risk Explorer")
st.caption(
    "Explore lending decisions, understand predictions, and inspect the evidence behind a model."
)
try:
    artifact_path = active_bundle_path()
    predictor, report, validation = load_run(str(artifact_path), artifact_path.stat().st_mtime_ns)
except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
    st.info(
        "No verified model is available yet. Follow the README's synthetic demo or original-data setup, then restart this app."
    )
    st.code(str(exc))
    st.stop()

is_synthetic = report["dataset_kind"] == "synthetic"
if is_synthetic:
    st.warning(
        "SYNTHETIC DEMO · Applicants and outcomes are generated for software demonstration. These results do not measure real credit performance."
    )
else:
    st.info(
        "Historical research results · Financial outcomes are simulations under stated assumptions, not realized revenue."
    )

st.sidebar.title("Explore")
page = st.sidebar.radio(
    "View",
    ["Policy explorer", "Applicant scoring", "Model quality", "Monitoring", "Project evidence"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("ACTIVE MODEL")
st.sidebar.write(report["selected_model"].replace("_", " ").title())
st.sidebar.caption("Synthetic demonstration" if is_synthetic else "Historical applicant data")
st.sidebar.caption(f"Version: {report['model_version']}")
st.sidebar.caption("CU = the input dataset's currency units.")
selected_name = report["selected_model"]
frozen_policy = DecisionPolicy(**report["policy"])

if page == "Policy explorer":
    st.subheader("What changes when the lending policy changes?")
    st.caption(
        "Explore the validation split here. The final test results remain frozen in Model quality. Controls do not change the API's active policy."
    )
    controls = st.columns(3)
    threshold = controls[0].slider(
        "Approve below this default probability",
        0.0,
        1.0,
        float(frozen_policy.threshold),
        0.005,
        format="%.3f",
    )
    margin = controls[1].slider(
        "Margin if repaid", 0.0, 0.40, float(frozen_policy.margin), 0.01, format="%.2f"
    )
    lgd = controls[2].slider(
        "Loss fraction if defaulted", 0.0, 1.0, float(frozen_policy.lgd), 0.01, format="%.2f"
    )
    scenario = DecisionPolicy(margin, lgd, threshold)
    result = portfolio_summary(
        validation.true_label, validation.pred_prob, validation.loan_amount, scenario
    )
    cols = st.columns(4)
    cols[0].metric("Approval rate", probability(result["approval_rate"]))
    cols[1].metric("Default rate among approved", probability(result["observed_default_rate"]))
    cols[2].metric("Simulated profit", currency(result["simulated_profit"]))
    cols[3].metric("Approved loan principal", currency(result["approved_exposure"]))
    st.caption(
        "Simulated profit uses observed validation defaults and your margin/loss assumptions. It excludes funding costs, prepayments, and loan-duration differences."
    )
    curves = []
    for cutoff in np.linspace(0, 1, 81):
        row = portfolio_summary(
            validation.true_label,
            validation.pred_prob,
            validation.loan_amount,
            DecisionPolicy(margin, lgd, float(cutoff)),
        )
        curves.append(
            {
                "Approval threshold": cutoff,
                "Simulated profit (CU)": row["simulated_profit"],
                "Approval rate": row["approval_rate"],
            }
        )
    left, right = st.columns([1.25, 1])
    fig = px.line(
        pd.DataFrame(curves),
        x="Approval threshold",
        y="Simulated profit (CU)",
        color_discrete_sequence=[ACCENT],
    )
    fig.add_scatter(
        x=[threshold],
        y=[result["simulated_profit"]],
        mode="markers",
        marker={"size": 12, "color": "#d97706"},
        name="Current scenario",
    )
    fig.update_layout(height=370, margin={"t": 25, "b": 25})
    left.plotly_chart(fig, use_container_width=True)
    right.subheader("Policy comparison")
    baseline_threshold = report["validation"]["logistic_baseline"]["policy"]["threshold"]
    baseline = portfolio_summary(
        validation.true_label,
        validation.baseline_prob,
        validation.loan_amount,
        DecisionPolicy(margin, lgd, baseline_threshold),
    )
    comparisons = []
    for name, summary in [
        ("Your scenario", result),
        ("Logistic baseline · saved threshold", baseline),
        (
            "Approve everyone",
            portfolio_summary(
                validation.true_label,
                validation.pred_prob,
                validation.loan_amount,
                scenario,
                approve_all=True,
            ),
        ),
    ]:
        comparisons.append(
            {
                "Policy": name,
                "Approval rate": probability(summary["approval_rate"]),
                "Simulated profit": currency(summary["simulated_profit"]),
            }
        )
    right.dataframe(pd.DataFrame(comparisons), hide_index=True, use_container_width=True)
    right.caption(
        "All policies use the same validation applicants and current financial assumptions. The logistic threshold stays at its original validated setting."
    )
    with st.expander("Stress scenario: what if predicted risk rises?", expanded=False):
        shock = st.slider(
            "Absolute increase in default probability", 0.0, 0.20, 0.02, 0.005, format="%.3f"
        )
        stress = stress_summary(validation.pred_prob, validation.loan_amount, scenario, shock)
        a, b, c = st.columns(3)
        a.metric("Baseline expected profit", currency(stress["baseline_expected_profit"]))
        b.metric(
            "Already funded loans · stressed",
            currency(stress["existing_portfolio_stressed_expected_profit"]),
        )
        c.metric(
            "Future applications · stressed",
            currency(stress["new_originations_stressed_expected_profit"]),
        )
        st.caption(
            "Already funded loans remain in the portfolio. Future applications may be declined under the frozen cutoff. This is a hypothetical probability shock; observed labels are not changed."
        )
    st.download_button(
        "Download this validation scenario",
        json.dumps(result, indent=2),
        "validation_policy_scenario.json",
        "application/json",
    )

elif page == "Applicant scoring":
    st.subheader("Inspect an applicant decision")
    st.caption(
        "The form starts with an illustrative profile built from training-set summaries. Review the additional fields before scoring. The API applies the saved policy."
    )
    example = predictor.metadata()["example_request"]["features"]
    labels = {
        "AMT_INCOME_TOTAL": "Annual income (CU)",
        "AMT_CREDIT": "Requested loan (CU)",
        "AMT_ANNUITY": "Loan annuity (CU)",
        "bureau_active_loans": "Active bureau loans",
        "bureau_total_loans": "Total bureau loans",
    }
    primary = [name for name in labels if name in example]
    with st.form("applicant"):
        columns = st.columns(3)
        edited = {}
        for index, name in enumerate(primary):
            value = example[name] if example[name] is not None else 0
            if name.startswith("bureau_"):
                edited[name] = columns[index % 3].number_input(
                    labels[name], min_value=0, value=int(value), step=1
                )
            else:
                edited[name] = columns[index % 3].number_input(
                    labels[name], min_value=0.01, value=max(float(value), 0.01), step=100.0
                )
        with st.expander("Additional applicant profile fields", expanded=False):
            additional = st.text_area(
                "Additional fields (JSON)",
                json.dumps(
                    {name: value for name, value in example.items() if name not in primary},
                    indent=2,
                ),
                height=230,
            )
        include_explanation = st.checkbox("Explain the prediction", value=True)
        submitted = st.form_submit_button("Score applicant", type="primary")
    if submitted:
        try:
            other = json.loads(additional)
            if not isinstance(other, dict) or set(other) & set(primary):
                raise ValueError(
                    "Additional fields must be an object without duplicates of the main form fields"
                )
            response = requests.post(
                API_URL + "/predict",
                json={"features": {**other, **edited}, "explain": include_explanation},
                timeout=60,
            )
            if response.status_code != 200:
                raise ValueError(response.json().get("detail", "API request failed"))
            prediction = response.json()
            if prediction["model_version"] != report["model_version"]:
                raise ValueError(
                    "The API and dashboard have different model versions. Restart both with the same model directory."
                )
            st.session_state["last_prediction"] = prediction
        except (ValueError, requests.RequestException) as exc:
            st.session_state.pop("last_prediction", None)
            st.error(str(exc))
    if prediction := st.session_state.get("last_prediction"):
        a, b, c = st.columns(3)
        a.metric("Predicted default probability", probability(prediction["predicted_pd"]))
        b.metric("Policy recommendation", prediction["decision"])
        c.metric("Expected profit if funded", currency(prediction["expected_profit_if_funded"]))
        st.caption(
            f"Saved approval cutoff: {prediction['threshold_used']:.2%}. Expected profit is assumption-based and does not guarantee a return."
        )
        if prediction["explanation"]:
            factors = pd.DataFrame(prediction["top_risk_factors"]).sort_values("contribution")
            fig = px.bar(
                factors,
                x="contribution",
                y="feature",
                orientation="h",
                color="direction",
                color_discrete_map={
                    "increases risk": "#dc6b54",
                    "decreases risk": ACCENT,
                    "no contribution": "#9ca3af",
                },
            )
            fig.update_layout(
                xaxis_title="SHAP contribution · base-model log odds",
                yaxis_title="",
                height=380,
                margin={"t": 20},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(prediction["explanation"]["note"])
    st.divider()
    st.subheader("Score a CSV batch")
    st.caption(
        "Up to 100 applicants. Use the template's input columns; financial ratios are computed by the service."
    )
    st.download_button(
        "Download input template",
        pd.DataFrame([example]).to_csv(index=False),
        "applicant_template.csv",
        "text/csv",
    )
    upload = st.file_uploader("Applicant CSV", type=["csv"])
    if upload is not None and st.button("Score batch"):
        try:
            frame = pd.read_csv(upload)
            applicants = json.loads(frame.to_json(orient="records"))
            response = requests.post(
                API_URL + "/predict/batch",
                json={"applicants": applicants, "explain": False},
                timeout=60,
            )
            if response.status_code != 200:
                raise ValueError(response.json().get("detail", "Batch request failed"))
            payload = response.json()
            if payload["model_version"] != report["model_version"]:
                raise ValueError("API and dashboard model versions differ")
            rows = [
                {
                    key: row[key]
                    for key in [
                        "request_id",
                        "predicted_pd",
                        "decision",
                        "loan_amount",
                        "expected_profit_under_policy",
                        "model_version",
                    ]
                }
                for row in payload["predictions"]
            ]
            st.session_state["batch_results"] = pd.DataFrame(rows)
        except (ValueError, requests.RequestException, pd.errors.ParserError) as exc:
            st.session_state.pop("batch_results", None)
            st.error(str(exc))
    if "batch_results" in st.session_state:
        st.dataframe(st.session_state["batch_results"], hide_index=True, use_container_width=True)
        st.download_button(
            "Download batch decisions",
            st.session_state["batch_results"].to_csv(index=False),
            "batch_decisions.csv",
            "text/csv",
        )

elif page == "Model quality":
    st.subheader("Final test results")
    st.caption(
        "The model and approval threshold were frozen before these test predictions were generated. Use validation data for any new policy exploration."
    )
    metrics = report["test"][selected_name]["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    b.metric("Average precision", f"{metrics['average_precision']:.3f}")
    c.metric("Brier score · lower is better", f"{metrics['brier_score']:.3f}")
    d.metric("Test default prevalence", probability(metrics["default_prevalence"]))
    rows = []
    for name, result in report["test"].items():
        row = {
            "Model / policy": name,
            **result.get("metrics", {}),
            "approval_rate": result["portfolio"]["approval_rate"],
            "simulated_profit": result["portfolio"]["simulated_profit"],
        }
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.plotly_chart(
        calibration_figure(
            {name: result for name, result in report["test"].items() if "calibration" in result}
        ),
        use_container_width=True,
    )
    st.caption(
        "Calibration compares average predicted probabilities with observed default frequencies. Sparse bins are noisy. Brier score measures overall probability quality, not calibration alone."
    )
    with st.expander("Candidate comparison · validation only"):
        rows = [
            {
                "Candidate": name,
                "Selected": name == selected_name,
                **result["metrics"],
                "simulated_profit": result["portfolio"]["simulated_profit"],
                "threshold": result["policy"]["threshold"],
            }
            for name, result in report["validation"].items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.plotly_chart(calibration_figure(report["validation"]), use_container_width=True)
    with st.expander("Loan-size cohort diagnostics"):
        st.dataframe(
            pd.DataFrame(report["test_cohorts"]), hide_index=True, use_container_width=True
        )
        st.caption("Descriptive diagnostics, not a demographic fairness assessment.")

elif page == "Monitoring":
    st.subheader("Are incoming prediction scores changing?")
    st.caption(
        "Only logs from this exact model version and dataset type are compared with its validation baseline."
    )
    try:
        drift = drift_report(artifact_path.parent, LOG_PATH)
        if drift["status"] == "insufficient_data":
            st.info(
                f"Waiting for at least 20 predictions for this model version. Available: {drift['live_samples']}."
            )
        else:
            a, b, c = st.columns(3)
            a.metric("Score PSI", f"{drift['psi']:.3f}")
            b.metric("Logged predictions", drift["live_samples"])
            c.metric("Other versions excluded", drift["excluded_other_versions"])
            if drift["status"] == "review_shift":
                st.warning(
                    "The score distribution has shifted. Investigate population and input changes before changing the model."
                )
            else:
                st.success("No score-distribution flag at the configured heuristic threshold.")
            st.caption(drift["note"])
    except (ValueError, KeyError, OSError) as exc:
        st.error(f"Monitoring inputs could not be read: {exc}")
    st.write(
        "Accuracy and default-rate monitoring require later repayment outcomes. Current API logs contain prediction metadata, not those outcomes."
    )

else:
    st.subheader("Reproducibility and scope")
    st.write(
        "This run records the dataset fingerprint, applicant split assignments, software versions, policy, validation comparisons, and final test results."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Split": report["split_counts"].keys(),
                "Applicants": report["split_counts"].values(),
                "Default rate": report["split_default_rates"].values(),
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.write(
        f"Input features: {report['features']}. Applicant identifiers were excluded from the model."
    )
    for limitation in report["limitations"]:
        st.write("• " + limitation)
    st.download_button(
        "Download evaluation report",
        json.dumps(report, indent=2),
        "credit_risk_evaluation.json",
        "application/json",
    )
    st.caption(f"Source SHA-256: {report['source_sha256']}")

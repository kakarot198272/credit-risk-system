"""Train candidates, calibrate separately, select on validation, test once."""

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.models.data import (
    AMOUNT_COLUMN,
    ID_COLUMN,
    TARGET,
    feature_frame,
    feature_schema,
    normalize_feature_types,
    read_dataset,
    split_applicants,
)
from src.risk.artifacts import ROOT, file_sha256, register_candidate, write_json
from src.risk.metrics import calibration_bins, classification_metrics, cohort_report
from src.risk.policy import DecisionPolicy, optimize_policy, portfolio_summary


def make_preprocessor(schema, scale=False):
    numerical = [name for name, spec in schema.items() if spec["type"] == "number"]
    categorical = [name for name, spec in schema.items() if spec["type"] == "category"]
    num_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numerical),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )


def transformed_feature_sources(preprocessor, schema):
    numerical = [name for name, spec in schema.items() if spec["type"] == "number"]
    categorical = [name for name, spec in schema.items() if spec["type"] == "category"]
    sources = numerical.copy()
    if categorical:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        for name, values in zip(categorical, encoder.categories_):
            sources.extend([name] * len(values))
    return sources


def train_experiment(
    data_path, model_dir, *, dataset_kind="real", activate=False, seed=42, quick=False
):
    data_path, model_dir = Path(data_path), Path(model_dir)
    if dataset_kind not in {"real", "synthetic"}:
        raise ValueError("dataset_kind must be real or synthetic")
    if "stressed" in data_path.stem.lower():
        raise ValueError(
            "Artificially relabeled stress data is not eligible for model training/evaluation"
        )
    frame = read_dataset(data_path)
    synthetic_marker = "__synthetic__" in frame and frame["__synthetic__"].fillna(False).any()
    if synthetic_marker and dataset_kind != "synthetic":
        raise ValueError("Synthetic dataset requires explicit --dataset-kind synthetic")
    if dataset_kind == "synthetic" and model_dir.resolve() == (ROOT / "models").resolve():
        raise ValueError(
            "Synthetic artifacts must use a separate model directory, e.g. artifacts/demo/models"
        )
    splits = split_applicants(frame, seed)
    economics = DecisionPolicy(**json.loads((ROOT / "src/config/decision_policy.json").read_text()))
    schema = feature_schema(feature_frame(splits["train"]))
    X = {
        name: normalize_feature_types(feature_frame(part), schema) for name, part in splits.items()
    }
    y = {name: part[TARGET].to_numpy(dtype=int) for name, part in splits.items()}
    amounts = {name: part[AMOUNT_COLUMN].to_numpy(dtype=float) for name, part in splits.items()}
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = model_dir / version
    run_dir.mkdir(parents=True, exist_ok=False)
    split_manifest = pd.concat(
        [
            pd.DataFrame({"applicant_id": part[ID_COLUMN], "split": name})
            for name, part in splits.items()
        ],
        ignore_index=True,
    )
    split_manifest.to_csv(run_dir / "split_assignments.csv", index=False)

    logistic = Pipeline(
        [
            ("preprocessor", make_preprocessor(schema, scale=True)),
            ("classifier", LogisticRegression(max_iter=2000, C=0.5, random_state=seed)),
        ]
    )
    logistic.fit(X["train"], y["train"])
    models = {"logistic_baseline": logistic}
    base_models = {"logistic_baseline": logistic}
    model_params = {"logistic_baseline": {"C": 0.5, "max_iter": 2000}}
    configurations = [
        ("xgboost_shallow", 3, 60 if quick else 200),
        ("xgboost_deeper", 5, 90 if quick else 350),
    ]
    for name, depth, trees in configurations:
        params = dict(
            n_estimators=trees,
            max_depth=depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=2,
            random_state=seed,
        )
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(schema)),
                ("classifier", XGBClassifier(**params)),
            ]
        )
        pipeline.fit(X["train"], y["train"])
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
        calibrated.fit(X["calibration"], y["calibration"])
        models[name], models[name + "_calibrated"] = pipeline, calibrated
        base_models[name] = base_models[name + "_calibrated"] = pipeline
        model_params[name] = params
        model_params[name + "_calibrated"] = {
            **params,
            "calibration": "sigmoid, separate calibration split",
        }

    validation_results, policies, validation_probabilities = {}, {}, {}
    for name, model in models.items():
        probabilities = model.predict_proba(X["validation"])[:, 1]
        policy, summary, _ = optimize_policy(
            y["validation"],
            probabilities,
            amounts["validation"],
            margin=economics.margin,
            lgd=economics.lgd,
        )
        policies[name], validation_probabilities[name] = policy, probabilities
        validation_results[name] = {
            "metrics": classification_metrics(y["validation"], probabilities),
            "policy": policy.to_dict(),
            "portfolio": summary,
            "calibration": calibration_bins(y["validation"], probabilities),
        }
    selected_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["portfolio"]["simulated_profit"],
            -validation_results[name]["metrics"]["brier_score"],
            name == "logistic_baseline",
        ),
    )
    selected = models[selected_name]
    policy = policies[selected_name]
    # This file records the choice before any final test predictions are computed.
    selection = {
        "model_version": version,
        "selected_model": selected_name,
        "selected_on": "validation simulated profit; Brier score breaks ties",
        "policy": policy.to_dict(),
        "dataset_kind": dataset_kind,
    }
    write_json(run_dir / "selection.json", selection)
    test_results = {}
    for name in dict.fromkeys(["logistic_baseline", selected_name]):
        probabilities = models[name].predict_proba(X["test"])[:, 1]
        test_results[name] = {
            "metrics": classification_metrics(y["test"], probabilities),
            "policy": policies[name].to_dict(),
            "portfolio": portfolio_summary(
                y["test"], probabilities, amounts["test"], policies[name]
            ),
            "calibration": calibration_bins(y["test"], probabilities),
        }
    selected_test_probabilities = selected.predict_proba(X["test"])[:, 1]
    test_results["approve_all"] = {
        "portfolio": portfolio_summary(
            y["test"], selected_test_probabilities, amounts["test"], policy, approve_all=True
        )
    }
    test_results["reject_all"] = {
        "portfolio": portfolio_summary(
            y["test"],
            selected_test_probabilities,
            amounts["test"],
            DecisionPolicy(margin=economics.margin, lgd=economics.lgd, threshold=0),
        )
    }
    val_predictions = pd.DataFrame(
        {
            "true_label": y["validation"],
            "loan_amount": amounts["validation"],
            "pred_prob": validation_probabilities[selected_name],
            "baseline_prob": validation_probabilities["logistic_baseline"],
        }
    )
    val_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(
        {
            "true_label": y["test"],
            "loan_amount": amounts["test"],
            "pred_prob": selected_test_probabilities,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)
    raw_pipeline = base_models[selected_name]
    transformed_background = raw_pipeline.named_steps["preprocessor"].transform(
        X["train"].iloc[:64]
    )
    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()
    bundle = {
        "schema_version": 2,
        "model_version": version,
        "dataset_kind": dataset_kind,
        "selected_model": selected_name,
        "model": selected,
        "base_pipeline": raw_pipeline,
        "policy": policy.to_dict(),
        "feature_schema": schema,
        "transformed_feature_sources": transformed_feature_sources(
            raw_pipeline.named_steps["preprocessor"], schema
        ),
        "explanation_background": transformed_background,
        "calibrated": selected_name.endswith("_calibrated"),
        "currency": "dataset currency units",
    }
    joblib.dump(bundle, run_dir / "bundle.joblib")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {
        **selection,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": data_path.name,
        "source_sha256": file_sha256(data_path),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "random_seed": seed,
        "features": len(schema),
        "split_counts": {name: len(part) for name, part in splits.items()},
        "split_default_rates": {name: float(part[TARGET].mean()) for name, part in splits.items()},
        "excluded_identifier_columns": [c for c in frame if c.startswith("SK_ID_")],
        "hyperparameters": model_params,
        "validation": validation_results,
        "test": test_results,
        "test_cohorts": cohort_report(
            y["test"], selected_test_probabilities, amounts["test"], policy.threshold
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "shap", "joblib"]
        },
        "limitations": [
            "Synthetic demo metrics are not evidence of real borrower performance."
            if dataset_kind == "synthetic"
            else "Results apply to this historical dataset and split, not a deployed lending population.",
            "Random applicant split; future-period generalization is not established.",
            "Profit is simulated using observed binary outcomes and fixed margin/LGD assumptions; no actual recoveries or cash flows are measured.",
            "Loan principal proxies exposure; funding costs, loan duration, prepayment and selection bias are not modeled.",
            "Loan-size cohort diagnostics are not a demographic fairness or regulatory compliance assessment.",
            "The test set is for final reporting; repeated tuning against its results invalidates the holdout.",
        ],
    }
    write_json(run_dir / "report.json", report)
    manifest_files = [
        "bundle.joblib",
        "selection.json",
        "report.json",
        "split_assignments.csv",
        "validation_predictions.csv",
        "test_predictions.csv",
    ]
    write_json(
        run_dir / "manifest.json",
        {"sha256": {name: file_sha256(run_dir / name) for name in manifest_files}},
    )
    register_candidate(model_dir, f"{version}/bundle.joblib", activate=activate)
    print(
        json.dumps(
            {
                "run_directory": str(run_dir),
                "dataset_kind": dataset_kind,
                "selected_model": selected_name,
                "activated": activate,
                "test_metrics": test_results[selected_name]["metrics"],
            },
            indent=2,
        )
    )
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/features/model_input.parquet")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--dataset-kind", choices=["real", "synthetic"], default="real")
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Explicitly set this locally trained candidate as active",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Smaller tree counts for demo/testing")
    args = parser.parse_args()
    train_experiment(
        args.data,
        args.model_dir,
        dataset_kind=args.dataset_kind,
        activate=args.activate,
        seed=args.seed,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()

"""Validated inference and SHAP explanations shared by CLI and FastAPI."""

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import DERIVED_DEPENDENCIES, engineer_financial_ratios
from src.models.data import normalize_feature_types
from src.risk.artifacts import active_bundle_path, file_sha256
from src.risk.policy import DecisionPolicy, expected_profit_if_funded


class InputError(ValueError):
    """Applicant inputs violated the trained model's feature contract."""


class Predictor:
    def __init__(self, artifact_path):
        self.artifact_path = Path(artifact_path)
        manifest_path = self.artifact_path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Artifact manifest missing; use a version-2 training run")
        manifest = json.loads(manifest_path.read_text())
        expected_hash = manifest["sha256"].get(self.artifact_path.name)
        if file_sha256(self.artifact_path) != expected_hash:
            raise ValueError("Artifact checksum mismatch")
        # Only load locally generated/trusted joblib artifacts: pickle can execute code.
        self.bundle = joblib.load(self.artifact_path)
        if self.bundle.get("schema_version") != 2:
            raise ValueError(
                "Unsupported artifact version; rebuild with the current training pipeline"
            )
        self.model = self.bundle["model"]
        self.policy = DecisionPolicy(**self.bundle["policy"])
        self.schema = self.bundle["feature_schema"]
        self.input_schema = {
            name: spec for name, spec in self.schema.items() if name not in DERIVED_DEPENDENCIES
        }
        self._explainer = None
        self._explanation_lock = threading.Lock()

    @classmethod
    def from_directory(cls, directory=None):
        return cls(active_bundle_path(directory))

    @property
    def version(self):
        return self.bundle["model_version"]

    def metadata(self):
        return {
            "model_version": self.version,
            "selected_model": self.bundle["selected_model"],
            "dataset_kind": self.bundle["dataset_kind"],
            "policy": self.policy.to_dict(),
            "calibrated": self.bundle["calibrated"],
            "currency": self.bundle["currency"],
            "input_features": self.input_schema,
            "derived_features": [name for name in self.schema if name in DERIVED_DEPENDENCIES],
            "example_request": {
                "features": {name: spec["example"] for name, spec in self.input_schema.items()},
                "explain": True,
            },
        }

    def prepare(self, applicants):
        rows = []
        for index, applicant in enumerate(applicants):
            if not isinstance(applicant, dict):
                raise InputError(f"Applicant {index}: features must be an object")
            missing = set(self.input_schema) - set(applicant)
            unknown = set(applicant) - set(self.input_schema)
            if missing or unknown:
                raise InputError(
                    f"Applicant {index}: missing fields {sorted(missing)}; unexpected fields {sorted(unknown)}"
                )
            row = {}
            for name, spec in self.input_schema.items():
                value = applicant[name]
                if value is None:
                    if not spec["nullable"]:
                        raise InputError(f"Applicant {index}: {name} cannot be null")
                    row[name] = np.nan
                elif spec["type"] == "number":
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                    ):
                        raise InputError(
                            f"Applicant {index}: {name} must be a finite number or null"
                        )
                    if name in {"AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"} and value <= 0:
                        raise InputError(f"Applicant {index}: {name} must be positive")
                    if name in {
                        "bureau_total_loans",
                        "bureau_active_loans",
                        "prev_app_count",
                        "prev_approved_count",
                    } and (value < 0 or value != int(value)):
                        raise InputError(f"Applicant {index}: {name} must be a nonnegative integer")
                    row[name] = float(value)
                elif not isinstance(value, str) or not value.strip() or len(value) > 200:
                    raise InputError(
                        f"Applicant {index}: {name} must be a nonempty string of at most 200 characters"
                    )
                else:
                    row[name] = value
            for part, total in [
                ("bureau_active_loans", "bureau_total_loans"),
                ("prev_approved_count", "prev_app_count"),
            ]:
                if (
                    part in row
                    and total in row
                    and pd.notna(row[part])
                    and pd.notna(row[total])
                    and row[part] > row[total]
                ):
                    raise InputError(f"Applicant {index}: {part} cannot exceed {total}")
            rows.append(row)
        frame = engineer_financial_ratios(pd.DataFrame(rows))
        return normalize_feature_types(frame.loc[:, list(self.schema)], self.schema)

    def explain(self, frame, top_k=8):
        import shap

        pipeline = self.bundle["base_pipeline"]
        transformed = pipeline.named_steps["preprocessor"].transform(frame)
        classifier = pipeline.named_steps["classifier"]
        # XGBoost distinguishes an absent sparse entry from an explicit zero.
        # Preserve the exact representation used by prediction for Tree SHAP.
        if not hasattr(classifier, "get_booster") and hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        with self._explanation_lock:
            if self._explainer is None:
                if hasattr(classifier, "get_booster"):
                    self._explainer = shap.TreeExplainer(classifier, model_output="raw")
                else:
                    self._explainer = shap.LinearExplainer(
                        classifier, self.bundle["explanation_background"]
                    )
            values = np.asarray(self._explainer.shap_values(transformed))[0]
            expected = float(np.asarray(self._explainer.expected_value).reshape(-1)[0])
        sources = self.bundle["transformed_feature_sources"]
        if len(values) != len(sources):
            raise RuntimeError("Explanation feature mapping is inconsistent with the artifact")
        contributions = {}
        for name, value in zip(sources, values):
            contributions[name] = contributions.get(name, 0.0) + float(value)
        ranked = sorted(contributions.items(), key=lambda pair: abs(pair[1]), reverse=True)
        factors = [
            {
                "feature": name,
                "contribution": effect,
                "direction": "increases risk"
                if effect > 0
                else "decreases risk"
                if effect < 0
                else "no contribution",
            }
            for name, effect in ranked[:top_k]
        ]
        return {
            "method": "SHAP",
            "scale": "base_model_log_odds",
            "base_value": expected,
            "total_contribution": float(sum(contributions.values())),
            "shown_contribution": float(sum(item["contribution"] for item in factors)),
            "factors": factors,
            "note": "Feature contributions explain the base model, not causal effects. They do not add directly to a calibrated probability.",
        }

    def predict_many(self, applicants, explain=False):
        frame = self.prepare(applicants)
        probabilities = self.model.predict_proba(frame)[:, 1]
        results = []
        for index, probability in enumerate(probabilities):
            probability = float(probability)
            amount = float(frame.iloc[index]["AMT_CREDIT"])
            approve = probability < self.policy.threshold
            expected_profit = float(expected_profit_if_funded(probability, amount, self.policy))
            explanation = self.explain(frame.iloc[[index]]) if explain else None
            results.append(
                {
                    "request_id": str(uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_version": self.version,
                    "dataset_kind": self.bundle["dataset_kind"],
                    "predicted_pd": probability,
                    "decision": "APPROVE" if approve else "REJECT",
                    "threshold_used": self.policy.threshold,
                    "loan_amount": amount,
                    "expected_profit_if_funded": expected_profit,
                    "expected_profit_under_policy": expected_profit if approve else 0.0,
                    "currency": self.bundle["currency"],
                    "explanation": explanation,
                    "top_risk_factors": explanation["factors"] if explanation else [],
                }
            )
        return results

    def predict(self, features, explain=True):
        return self.predict_many([features], explain=explain)[0]

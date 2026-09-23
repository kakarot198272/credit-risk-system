import json

import numpy as np
import pandas as pd
import pytest

from src.models.data import feature_frame, normalize_feature_types, split_applicants
from src.models.train import train_experiment


def test_applicants_are_disjoint_and_identifiers_are_excluded(trained_run):
    assignments = pd.read_csv(trained_run["run"] / "split_assignments.csv")
    assert assignments.applicant_id.is_unique
    assert set(assignments.applicant_id) == set(trained_run["frame"].SK_ID_CURR)
    assert set(assignments.split) == {"train", "calibration", "validation", "test"}
    assert not any(
        name.startswith("SK_ID_") or name in {"TARGET", "__synthetic__"}
        for name in trained_run["predictor"].schema
    )


def test_imputer_is_fitted_only_on_training_applicants(trained_run):
    predictor = trained_run["predictor"]
    train = split_applicants(trained_run["frame"])["train"]
    features = normalize_feature_types(feature_frame(train), predictor.schema)
    numerical = [name for name, spec in predictor.schema.items() if spec["type"] == "number"]
    imputer = (
        predictor.bundle["base_pipeline"]
        .named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps["imputer"]
    )
    np.testing.assert_allclose(
        imputer.statistics_, features[numerical].median().to_numpy(), equal_nan=True
    )


def test_changing_only_test_labels_does_not_change_selection(trained_run, tmp_path, monkeypatch):
    import src.models.train as module

    original_split = module.split_applicants

    def inverted_test_labels(frame, seed):
        parts = original_split(frame, seed)
        parts["test"]["TARGET"] = 1 - parts["test"]["TARGET"]
        return parts

    monkeypatch.setattr(module, "split_applicants", inverted_test_labels)
    altered_run = train_experiment(
        trained_run["data"], tmp_path / "models", dataset_kind="synthetic", quick=True
    )
    original = json.loads((trained_run["run"] / "report.json").read_text())
    altered = json.loads((altered_run / "report.json").read_text())
    assert original["selected_model"] == altered["selected_model"]
    assert original["policy"] == altered["policy"]
    assert original["validation"] == altered["validation"]
    name = original["selected_model"]
    assert (
        original["test"][name]["metrics"]["roc_auc"] != altered["test"][name]["metrics"]["roc_auc"]
    )
    registry = json.loads((tmp_path / "models/registry.json").read_text())
    assert not registry.get("production_model")


def test_repeated_applicants_are_rejected(trained_run):
    duplicate = pd.concat([trained_run["frame"], trained_run["frame"].iloc[[0]]])
    with pytest.raises(ValueError, match="exactly once"):
        split_applicants(duplicate)


def test_synthetic_and_stressed_data_cannot_be_mislabeled(trained_run, tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        train_experiment(trained_run["data"], tmp_path / "models")
    with pytest.raises(ValueError, match="stress"):
        train_experiment(tmp_path / "model_input_stressed.parquet", tmp_path / "models")

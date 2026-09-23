import json

import pytest
from fastapi.testclient import TestClient

from src.inference.api import create_app
from src.inference.predict import Predictor
from src.models.data import feature_frame, normalize_feature_types


@pytest.fixture
def client(trained_run, tmp_path):
    with TestClient(create_app(trained_run["directory"], tmp_path / "audit.jsonl")) as client:
        yield client


def test_single_and_batch_match_offline_predictions(client, trained_run):
    predictor = trained_run["predictor"]
    row = trained_run["frame"].iloc[[0]]
    features = normalize_feature_types(feature_frame(row), predictor.schema)
    expected = float(predictor.model.predict_proba(features)[0, 1])
    payload = json.loads(row[list(predictor.input_schema)].to_json(orient="records"))[0]
    single = client.post("/predict", json={"features": payload, "explain": False})
    batch = client.post("/predict/batch", json={"applicants": [payload], "explain": False})
    assert single.status_code == batch.status_code == 200
    assert single.json()["predicted_pd"] == pytest.approx(expected, abs=1e-7)
    assert single.json()["predicted_pd"] == batch.json()["predictions"][0]["predicted_pd"]


def test_shap_is_consistent_with_base_model_and_has_correct_units(client, applicant, trained_run):
    response = client.post("/predict", json={"features": applicant})
    assert response.status_code == 200
    explanation = response.json()["explanation"]
    predictor = trained_run["predictor"]
    base = predictor.bundle["base_pipeline"]
    transformed = base.named_steps["preprocessor"].transform(predictor.prepare([applicant]))
    classifier = base.named_steps["classifier"]
    if hasattr(classifier, "get_booster"):
        expected = float(classifier.predict(transformed, output_margin=True)[0])
    else:
        expected = float(classifier.decision_function(transformed)[0])
    assert explanation["base_value"] + explanation["total_contribution"] == pytest.approx(
        expected, abs=1e-5
    )
    assert explanation["scale"] == "base_model_log_odds"
    assert response.json()["dataset_kind"] == "synthetic"


@pytest.mark.parametrize(
    "mutation", ["missing", "unknown", "negative", "wrong_type", "derived", "inconsistent_counts"]
)
def test_bad_applicants_return_422(client, applicant, mutation):
    if mutation == "missing":
        applicant.pop("AMT_CREDIT")
    elif mutation == "unknown":
        applicant["TARGET"] = 1
    elif mutation == "negative":
        applicant["AMT_CREDIT"] = -10
    elif mutation == "wrong_type":
        applicant["AMT_CREDIT"] = "1000"
    elif mutation == "derived":
        applicant["credit_to_income_ratio"] = 999
    else:
        applicant["bureau_active_loans"] = applicant["bureau_total_loans"] + 1
    assert client.post("/predict", json={"features": applicant}).status_code == 422


def test_rejects_nonfinite_json_and_oversized_batch(client, applicant):
    applicant["AMT_CREDIT"] = float("inf")
    response = client.post(
        "/predict",
        content=json.dumps({"features": applicant}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    applicant["AMT_CREDIT"] = 100
    assert client.post("/predict/batch", json={"applicants": [applicant] * 101}).status_code == 422


def test_logs_exclude_applicant_features(trained_run, applicant, tmp_path):
    log = tmp_path / "audit.jsonl"
    with TestClient(create_app(trained_run["directory"], log)) as client:
        result = client.post("/predict", json={"features": applicant, "explain": False})
    assert result.status_code == 200
    entry = json.loads(log.read_text())
    assert entry["model_version"] == trained_run["predictor"].version
    assert not set(applicant) & set(entry)


def test_no_model_is_reported_as_not_ready(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["ready"] is False
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"features": {}}).status_code == 503


def test_tampered_model_is_rejected_before_unpickling(trained_run, tmp_path):
    artifact = tmp_path / "bundle.joblib"
    artifact.write_bytes(b"not a valid model")
    (tmp_path / "manifest.json").write_text((trained_run["run"] / "manifest.json").read_text())
    with pytest.raises(ValueError, match="checksum"):
        Predictor(artifact)

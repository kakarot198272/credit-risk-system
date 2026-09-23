from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.inference.api import create_app
from src.risk.artifacts import ROOT


def test_dashboard_views_and_policy_controls(trained_run, monkeypatch, tmp_path):
    monkeypatch.setenv("CREDIT_RISK_MODEL_DIR", str(trained_run["directory"]))
    monkeypatch.setenv("CREDIT_RISK_LOG_PATH", str(tmp_path / "missing.jsonl"))
    app = AppTest.from_file(str(ROOT / "src/dashboard/app.py"), default_timeout=45).run()
    assert not app.exception
    assert "SYNTHETIC" in app.warning[0].value
    app.slider[0].set_value(0.0).run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label == "Approval rate") == "0.0%"
    assert next(m.value for m in app.metric if m.label == "Simulated profit") == "0 CU"
    for page in ["Applicant scoring", "Model quality", "Monitoring", "Project evidence"]:
        app.radio[0].set_value(page).run()
        assert not app.exception, page


def test_applicant_form_reaches_api(trained_run, monkeypatch, tmp_path):
    import requests

    monkeypatch.setenv("CREDIT_RISK_MODEL_DIR", str(trained_run["directory"]))
    with TestClient(create_app(trained_run["directory"], tmp_path / "audit.jsonl")) as client:
        monkeypatch.setattr(
            requests, "post", lambda url, json, timeout: client.post("/predict", json=json)
        )
        app = AppTest.from_file(str(ROOT / "src/dashboard/app.py"), default_timeout=45).run()
        app.radio[0].set_value("Applicant scoring").run()
        next(button for button in app.button if button.label == "Score applicant").click().run()
        assert not app.exception
        assert not app.error
        assert any(metric.label == "Predicted default probability" for metric in app.metric)

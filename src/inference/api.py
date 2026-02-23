import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REGISTRY_PATH = "models/registry.json"
POLICY_PATH = "src/config/decision_policy.json"
LOG_PATH = "logs/prediction_logs.jsonl"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Production Model
# -----------------------------
def load_production_model():
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    model_file = registry.get("production_model")
    if not model_file:
        raise RuntimeError("No production model found in registry.")

    model_path = os.path.join("models", model_file)
    model = joblib.load(model_path)

    return model, model_file


def load_policy():
    with open(POLICY_PATH, "r") as f:
        return json.load(f)


def log_prediction(entry: dict):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# -----------------------------
# Business Helpers
# -----------------------------
def assign_bucket(p: float) -> str:
    if p < 0.05:
        return "0-5%"
    elif p < 0.10:
        return "5-10%"
    elif p < 0.15:
        return "10-15%"
    elif p < 0.20:
        return "15-20%"
    else:
        return "20%+"


def compute_expected_profit(p: float, margin: float, lgd: float) -> float:
    return (1 - p) * margin - p * lgd


def safe_jsonable(val: Any):
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    return val


# -----------------------------
# Startup
# -----------------------------
model, model_version = load_production_model()
policy = load_policy()

MARGIN = float(policy["margin"])
LGD = float(policy["lgd"])
THRESHOLD = float(policy["threshold"])

# Explicitly disable SHAP in production
shap_bundle = {"enabled": False}

try:
    expected_raw_cols = model.feature_names_in_.tolist()
except Exception:
    expected_raw_cols = None


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def health():
    return {
        "status": "Credit Risk API running",
        "model_version": model_version,
        "shap_enabled": False
    }


@app.post("/predict")
def predict(data: dict):
    try:
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        df = pd.DataFrame([data])

        if expected_raw_cols:
            df = df.reindex(columns=expected_raw_cols)

        prob = float(model.predict_proba(df)[0][1])

        decision = "APPROVE" if prob < THRESHOLD else "REJECT"
        expected_profit = compute_expected_profit(prob, MARGIN, LGD)
        bucket = assign_bucket(prob)

        response = {
            "request_id": request_id,
            "model_version": model_version,
            "predicted_pd": round(prob, 4),
            "decision": decision,
            "expected_profit": round(expected_profit, 4),
            "risk_bucket": bucket,
            "threshold_used": THRESHOLD,
            "timestamp": timestamp,
            "top_risk_factors": []
        }

        log_prediction({k: safe_jsonable(v) for k, v in response.items()})

        return response

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
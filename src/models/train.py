import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.mlops.experiment_logger import log_experiment

FEATURE_PATH = "data/features"
REGISTRY_PATH = "models/registry.json"

MARGIN = 0.10
LGD = 0.60
THRESHOLDS = np.linspace(0.01, 0.5, 200)


# ---------------------------------------------------
# PROFIT FUNCTIONS
# ---------------------------------------------------
def compute_profit(probs, threshold):
    approve = probs < threshold
    approved_probs = probs[approve]

    expected_loss = np.sum(approved_probs * LGD)
    expected_revenue = np.sum((1 - approved_probs) * MARGIN)
    expected_profit = expected_revenue - expected_loss

    return expected_profit


def optimize_threshold(probs):
    best_profit = -np.inf
    best_threshold = 0

    for t in THRESHOLDS:
        profit = compute_profit(probs, t)
        if profit > best_profit:
            best_profit = profit
            best_threshold = t

    return best_threshold, best_profit


def evaluate_model(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    threshold, profit = optimize_threshold(probs)
    return auc, profit


# ---------------------------------------------------
# REGISTRY HELPERS
# ---------------------------------------------------
def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {}

    try:
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠ Corrupted registry detected. Resetting registry.")
        return {}


def save_registry(registry):
    os.makedirs("models", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)


# ---------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------
def train_models():

    df = pd.read_parquet("data/features/model_input_stressed.parquet")

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = X.select_dtypes(include=["category"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # ---------------------------------------------------
    # Train Candidate Model
    # ---------------------------------------------------
    candidate_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    candidate_model.fit(X_train, y_train)

    candidate_probs = candidate_model.predict_proba(X_val)[:, 1]
    candidate_auc = roc_auc_score(y_val, candidate_probs)
    candidate_threshold, candidate_profit = optimize_threshold(candidate_probs)

    print("\n===== NEW CANDIDATE MODEL =====")
    print(f"AUC: {candidate_auc:.4f}")
    print(f"Profit: {candidate_profit:.2f}")

    # ---------------------------------------------------
    # Save Versioned Model
    # ---------------------------------------------------
    model_version = f"xgboost_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pkl"
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", model_version)
    joblib.dump(candidate_model, model_path)

    print(f"Model saved to {model_path}")

    # ---------------------------------------------------
    # Log Experiment
    # ---------------------------------------------------
    log_experiment(
        model_name="XGBoost",
        params={
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 8
        },
        auc=float(candidate_auc),
        profit=float(candidate_profit),
        threshold=float(candidate_threshold)
    )

    # ---------------------------------------------------
    # CHAMPION–CHALLENGER GOVERNANCE
    # ---------------------------------------------------
    registry = load_registry()

    production_version = registry.get("production_model")

    if production_version:
        print("\nEvaluating against current production model...")

        prod_model_path = os.path.join("models", production_version)
        prod_model = joblib.load(prod_model_path)

        prod_auc, prod_profit = evaluate_model(prod_model, X_val, y_val)

        print(f"Production AUC: {prod_auc:.4f}")
        print(f"Production Profit: {prod_profit:.2f}")

        print(f"Candidate AUC: {candidate_auc:.4f}")
        print(f"Candidate Profit: {candidate_profit:.2f}")

        # Promotion rule
        if candidate_profit > prod_profit and candidate_auc >= prod_auc - 0.01:
            print("\n🚀 Candidate outperforms production. Promoting.")
            registry["production_model"] = model_version
        else:
            print("\n❌ Candidate does NOT outperform production. Keeping production model.")

    else:
        print("\nNo production model found. Promoting candidate automatically.")
        registry["production_model"] = model_version

    # Always store latest candidate info
    registry["latest_candidate"] = model_version
    registry["latest_candidate_profit"] = float(candidate_profit)
    registry["latest_candidate_auc"] = float(candidate_auc)

    save_registry(registry)

    print("\nRegistry updated successfully.")


if __name__ == "__main__":
    train_models()
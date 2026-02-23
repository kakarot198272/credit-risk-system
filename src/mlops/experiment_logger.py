import json
import os
from datetime import datetime

EXPERIMENT_PATH = "artifacts/experiments/experiment_log.json"

def log_experiment(model_name, params, auc, profit, threshold):
    os.makedirs(os.path.dirname(EXPERIMENT_PATH), exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "parameters": params,
        "auc": float(auc),
        "profit": float(profit),
        "threshold": float(threshold)
    }

    if os.path.exists(EXPERIMENT_PATH):
        with open(EXPERIMENT_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(EXPERIMENT_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print("Experiment logged successfully.")

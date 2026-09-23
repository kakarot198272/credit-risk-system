import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

FEATURE_PATH = "data/features/model_input.parquet"

MARGIN = 0.10
LGD = 0.60


def calculate_profit(pred_probs, threshold):
    decisions = pred_probs < threshold
    profits = np.where(
        decisions,
        (1 - pred_probs) * MARGIN - pred_probs * LGD,
        0
    )
    return profits.sum()


def optimize_threshold(pred_probs):
    thresholds = np.linspace(0.01, 0.5, 200)

    best_profit = -np.inf
    best_threshold = 0

    for t in thresholds:
        profit = calculate_profit(pred_probs, t)
        if profit > best_profit:
            best_profit = profit
            best_threshold = t

    return best_threshold, best_profit


def main():
    df = pd.read_parquet(FEATURE_PATH)

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

    param_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300},
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 400},
        {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 500},
        {"max_depth": 5, "learning_rate": 0.07, "n_estimators": 250},
    ]

    best_profit = -np.inf
    best_params = None
    best_auc = 0
    best_accuracy = 0

    for params in param_grid:
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                **params
            ))
        ])

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]

        # Compute AUC
        auc = roc_auc_score(y_val, probs)

        # Optimize threshold for profit
        threshold, profit = optimize_threshold(probs)

        # Compute accuracy at optimal threshold
        pred_labels = (probs < threshold).astype(int)
        accuracy = accuracy_score(y_val, pred_labels)

        print(f"Params: {params}")
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy (at optimal threshold): {accuracy:.4f}")
        print(f"Profit: {profit:.2f}")
        print("-" * 50)

        if profit > best_profit:
            best_profit = profit
            best_params = params
            best_auc = auc
            best_accuracy = accuracy

    print("\n===== BEST CONFIGURATION =====")
    print(f"Best Params: {best_params}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Best Accuracy (at optimal threshold): {best_accuracy:.4f}")
    print(f"Best Profit: {best_profit:.2f}")


if __name__ == "__main__":
    main()

"""Data contracts and disjoint applicant splits before any fitted transforms."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import engineer_financial_ratios

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"
AMOUNT_COLUMN = "AMT_CREDIT"


def read_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {path}. Run feature building or the synthetic demo command."
        )
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def validate_dataset(frame):
    required = {TARGET, ID_COLUMN, AMOUNT_COLUMN, "AMT_INCOME_TOTAL", "AMT_ANNUITY"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Each applicant ID must occur exactly once; aggregate repeated records before splitting."
        )
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError(
            "TARGET must contain both 0 (no recorded default) and 1 (recorded default)"
        )
    if frame[TARGET].value_counts().min() < 20:
        raise ValueError("Need at least 20 applicants in each class for four stratified subsets")
    for column in (AMOUNT_COLUMN, "AMT_INCOME_TOTAL", "AMT_ANNUITY"):
        values = pd.to_numeric(frame[column], errors="coerce")
        observed = values.dropna()
        if (
            (column == AMOUNT_COLUMN and values.isna().any())
            or not np.isfinite(observed).all()
            or (observed <= 0).any()
        ):
            raise ValueError(
                f"{column} must be positive and finite when present; loan amounts cannot be missing"
            )


def split_applicants(frame, random_state=42):
    validate_dataset(frame)
    positions = np.arange(len(frame))
    train_cal_val, test = train_test_split(
        positions, test_size=0.20, stratify=frame[TARGET], random_state=random_state
    )
    train_cal, validation = train_test_split(
        train_cal_val,
        test_size=0.15 / 0.80,
        stratify=frame.iloc[train_cal_val][TARGET],
        random_state=random_state + 1,
    )
    train, calibration = train_test_split(
        train_cal,
        test_size=0.15 / 0.65,
        stratify=frame.iloc[train_cal][TARGET],
        random_state=random_state + 2,
    )
    return {
        name: frame.iloc[idx].copy()
        for name, idx in {
            "train": train,
            "calibration": calibration,
            "validation": validation,
            "test": test,
        }.items()
    }


def feature_frame(frame):
    excluded = [c for c in frame if c == TARGET or c.startswith("SK_ID_") or c.startswith("__")]
    return engineer_financial_ratios(frame.drop(columns=excluded)).replace(
        [np.inf, -np.inf], np.nan
    )


def feature_schema(train_features):
    schema = {}
    for name in train_features:
        column = train_features[name]
        categorical = not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(
            column
        )
        clean = column.dropna()
        if categorical:
            categories = sorted(clean.astype(str).unique().tolist())
            example = str(clean.astype(str).mode().iloc[0]) if len(clean) else "Unknown"
            schema[name] = {
                "type": "category",
                "nullable": True,
                "example": example,
                "known_categories": categories[:100],
            }
        else:
            example = float(clean.median()) if len(clean) else None
            if (
                name
                in {
                    "bureau_total_loans",
                    "bureau_active_loans",
                    "prev_app_count",
                    "prev_approved_count",
                }
                and example is not None
            ):
                example = int(example)
            schema[name] = {
                "type": "number",
                "nullable": name != "AMT_CREDIT",
                "example": example,
            }
    return schema


def normalize_feature_types(frame, schema):
    frame = frame.copy()
    for name, spec in schema.items():
        if spec["type"] == "category":
            frame[name] = (
                frame[name]
                .astype(object)
                .map(lambda value: str(value) if pd.notna(value) else np.nan)
            )
        else:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    return frame

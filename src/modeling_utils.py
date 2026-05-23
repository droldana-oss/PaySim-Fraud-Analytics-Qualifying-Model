"""Utilidades reproducibles para la Parte 2 del modelado de fraude PaySim."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

TARGET = "isFraud"
ALERT_COLUMN = "isFlaggedFraud"
TRAIN_END_STEP = 323
VALIDATION_END_STEP = 378
LEGITIMATE_TO_FRAUD_RATIO = 20
RANDOM_STATE = 42

RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    TARGET,
    ALERT_COLUMN,
]
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origin_balance_diff",
    "dest_balance_diff",
    "balance_error_origin",
    "balance_error_dest",
    "amount_to_origin_balance",
    "is_origin_balance_zero",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_transactions(data_path: Path) -> pd.DataFrame:
    """Load only columns required for modeling using memory-aware dtypes."""
    if not data_path.exists():
        raise FileNotFoundError(f"No encontré el dataset en {data_path}")
    dtype_map = {
        "step": "int16",
        "type": "category",
        TARGET: "int8",
        ALERT_COLUMN: "int8",
    }
    return pd.read_csv(data_path, usecols=RAW_COLUMNS, dtype=dtype_map)


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create explainable risk signals discovered during EDA."""
    missing = set(RAW_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    data = raw.copy()
    data["log_amount"] = np.log1p(data["amount"])
    data["origin_balance_diff"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["dest_balance_diff"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["balance_error_origin"] = data["oldbalanceOrg"] - data["amount"] - data["newbalanceOrig"]
    data["balance_error_dest"] = data["oldbalanceDest"] + data["amount"] - data["newbalanceDest"]
    data["amount_to_origin_balance"] = data["amount"] / (data["oldbalanceOrg"] + 1.0)
    data["is_origin_balance_zero"] = (data["newbalanceOrig"] == 0).astype("int8")
    return data


def temporal_split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve later steps for validation and final testing."""
    train = data.loc[data["step"] <= TRAIN_END_STEP].copy()
    validation = data.loc[
        (data["step"] > TRAIN_END_STEP) & (data["step"] <= VALIDATION_END_STEP)
    ].copy()
    test = data.loc[data["step"] > VALIDATION_END_STEP].copy()
    for label, frame in {"train": train, "validation": validation, "test": test}.items():
        if frame.empty or frame[TARGET].sum() == 0:
            raise ValueError(f"El periodo {label} quedó vacío o sin fraudes")
    return train, validation, test


def split_summary(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        frauds = int(frame[TARGET].sum())
        rows.append({
            "periodo": name,
            "transacciones": len(frame),
            "fraudes": frauds,
            "tasa_fraude_pct": round(frauds / len(frame) * 100, 4),
            "step_min": int(frame["step"].min()),
            "step_max": int(frame["step"].max()),
        })
    return pd.DataFrame(rows).set_index("periodo")


def training_sample(train: pd.DataFrame, ratio: int = LEGITIMATE_TO_FRAUD_RATIO) -> pd.DataFrame:
    """Keep every fraud and sample legitimate operations only in training."""
    fraud = train.loc[train[TARGET] == 1]
    legitimate = train.loc[train[TARGET] == 0]
    n_legitimate = min(len(legitimate), len(fraud) * ratio)
    legitimate_sample = legitimate.sample(n=n_legitimate, random_state=RANDOM_STATE)
    return pd.concat([fraud, legitimate_sample]).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)


def xy(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return frame[MODEL_FEATURES], frame[TARGET]


def build_random_forest_pipeline():
    """Build the single approved classifier and its categorical preprocessing."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    preprocessing = ColumnTransformer(
        transformers=[
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", classifier)])


def select_f2_threshold(y_true: pd.Series, probability: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """Select threshold using validation only, emphasizing recall with beta=2."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = (4 * precision) + recall
    f2 = np.divide(5 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    curve = pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall, "f2": f2})
    best_index = int(curve["f2"].idxmax())
    return float(curve.loc[best_index, "threshold"]), curve


def metric_summary(frame: pd.DataFrame, probability: np.ndarray, threshold: float, label: str) -> pd.Series:
    """Return fraud-focused metrics and economic recovery for a scoring rule."""
    from sklearn.metrics import average_precision_score, confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score

    y_true = frame[TARGET].to_numpy()
    prediction = (probability >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    fraud_amount = frame.loc[frame[TARGET] == 1, "amount"].sum()
    detected_amount = frame.loc[(frame[TARGET] == 1) & (prediction == 1), "amount"].sum()
    return pd.Series({
        "regla": label,
        "threshold": threshold,
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f2": fbeta_score(y_true, prediction, beta=2, zero_division=0),
        "average_precision": average_precision_score(y_true, probability),
        "roc_auc": roc_auc_score(y_true, probability),
        "verdaderos_positivos": int(tp),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_negativos": int(tn),
        "monto_fraude_detectado": float(detected_amount),
        "monto_fraude_total": float(fraud_amount),
        "monto_detectado_pct": float(detected_amount / fraud_amount * 100),
    })


def original_alert_summary(frame: pd.DataFrame) -> pd.Series:
    """Measure the original binary PaySim alert with the same business metrics."""
    alert = frame[ALERT_COLUMN].to_numpy(dtype="int8")
    return metric_summary(frame, alert.astype(float), 0.5, "Alerta original isFlaggedFraud")

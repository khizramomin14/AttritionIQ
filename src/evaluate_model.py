"""
evaluate_model.py
Evaluation helpers: ROC curve data, confusion matrix, model comparison table.
"""

import os
import json
import numpy as np
import joblib
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_dataset, prepare_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")


def load_metadata() -> dict:
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "model_metadata.json not found. Run src/train_model.py first."
        )
    with open(METADATA_PATH) as f:
        return json.load(f)


def get_model_comparison() -> list:
    """Returns a list of dicts suitable for the comparison table."""
    meta = load_metadata()
    best = meta["best_model"]
    rows = []
    for name, m in meta["metrics"].items():
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "is_best": name == best,
        })
    # Sort by combined score
    rows.sort(key=lambda r: (r["f1"] + r["recall"] + r["roc_auc"]) / 3, reverse=True)
    return rows


def get_roc_curve_data() -> dict:
    """Compute ROC curve for the saved best model on the test set."""
    from sklearn.metrics import roc_curve, auc

    pipeline = joblib.load(BEST_MODEL_PATH)
    df = load_dataset()
    _, X_test, _, y_test, _, _, _ = prepare_data(df)

    y_score = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": round(float(roc_auc), 4),
    }


def get_confusion_matrix_data() -> dict:
    """Return confusion matrix for the saved best model."""
    from sklearn.metrics import confusion_matrix

    pipeline = joblib.load(BEST_MODEL_PATH)
    df = load_dataset()
    _, X_test, _, y_test, _, _, _ = prepare_data(df)

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "matrix": cm.tolist(),
        "labels": ["Retained (0)", "Left (1)"],
    }


def get_feature_importance() -> list:
    """Return top feature importances from metadata."""
    meta = load_metadata()
    return meta.get("top_features", [])

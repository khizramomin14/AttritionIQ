"""
fairness.py
Fairness and bias audit module for AttritionIQ.

Computes per-group performance metrics (Precision, Recall, F1,
False Positive Rate, False Negative Rate) for diagnostic purposes.

IMPORTANT: Fairness metrics are diagnostic indicators and do not
by themselves establish discrimination.
"""

import os
import sys
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")


def compute_fairness_metrics(df, pipeline=None, meta=None) -> dict:
    """
    Compute per-group model performance metrics on the full dataset.

    Groups evaluated: Gender (primary), Department (secondary).
    Protected attributes are used ONLY for diagnostic fairness analysis,
    not for recommendations.
    """
    from data_preprocessing import prepare_data, CONSTANT_COLUMNS, ID_COLUMNS, TARGET_COLUMN

    if pipeline is None:
        pipeline = joblib.load(BEST_MODEL_PATH)
    if meta is None:
        with open(METADATA_PATH) as f:
            meta = json.load(f)

    # Prepare test split (same seed/size as training to get consistent test set)
    _, X_test, _, y_test, _, _, _ = prepare_data(df)

    # Get test-set indices to align with original df rows
    from sklearn.model_selection import train_test_split
    from data_preprocessing import RANDOM_STATE, TEST_SIZE

    df_copy = df.copy()
    df_copy["_attrition_bin"] = (df_copy[TARGET_COLUMN] == "Yes").astype(int)

    drop_cols = CONSTANT_COLUMNS + ID_COLUMNS + [TARGET_COLUMN]
    drop_cols = [c for c in drop_cols if c in df_copy.columns]
    X_all = df_copy.drop(columns=drop_cols + ["_attrition_bin"])
    y_all = df_copy["_attrition_bin"]

    X_train_idx, X_test_idx, _, _ = train_test_split(
        df_copy.index, y_all,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )
    test_df = df_copy.loc[X_test_idx].copy()
    y_true = test_df["_attrition_bin"].values
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "groups": {},
        "disclaimer": (
            "Fairness metrics are diagnostic indicators and do not by "
            "themselves establish discrimination."
        ),
        "note": (
            "Gender and other demographic attributes are used here ONLY for "
            "diagnostic analysis and are not used in HR recommendations."
        ),
    }

    def _metrics_for_mask(mask):
        if mask.sum() == 0:
            return None
        yt = y_true[mask]
        yp = y_pred[mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        attrition_rate = round(float(yt.mean()) * 100, 1)

        return {
            "n_records": int(mask.sum()),
            "attrition_rate": attrition_rate,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "fpr": round(fpr * 100, 1),
            "fnr": round(fnr * 100, 1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    # Evaluate per group column
    group_cols = ["Gender", "Department", "JobRole", "MaritalStatus"]
    for col in group_cols:
        if col not in test_df.columns:
            continue
        col_groups = {}
        for val in sorted(test_df[col].unique()):
            mask = (test_df[col] == val).values
            m = _metrics_for_mask(mask)
            if m:
                col_groups[str(val)] = m
        result["groups"][col] = col_groups

    # Overall
    overall_mask = np.ones(len(y_true), dtype=bool)
    result["overall"] = _metrics_for_mask(overall_mask)
    result["test_size"] = int(len(y_true))

    return result

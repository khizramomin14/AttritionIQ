"""
train_model.py
Trains multiple classification models, compares them, selects the best one
and saves it together with the preprocessing pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_dataset, prepare_data, get_feature_names_out

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ─── Model definitions ────────────────────────────────────────────────────────

def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
    }


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor):
    """Fit a full pipeline (preprocessor + model) and return metrics."""
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return pipe, metrics


# ─── Training entry point ─────────────────────────────────────────────────────

def train_and_select(verbose=True):
    """
    Load data → preprocess → train candidates → pick best → save.
    Returns the full results dict.
    """
    # 1. Load data
    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = prepare_data(df)

    if verbose:
        print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
        print(f"Class distribution (train): {dict(y_train.value_counts())}")

    # 2. Train every candidate
    candidates = get_candidate_models()
    results = {}
    best_score = -1
    best_name = None
    best_pipeline = None

    for name, model in candidates.items():
        if verbose:
            print(f"  Training {name}...")
        pipe, metrics = evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor)
        results[name] = metrics
        if verbose:
            print(f"    F1={metrics['f1']}  Recall={metrics['recall']}  ROC-AUC={metrics['roc_auc']}")

        # Selection criterion: equally weighted F1 + Recall + ROC-AUC
        combined = (metrics["f1"] + metrics["recall"] + metrics["roc_auc"]) / 3
        if combined > best_score:
            best_score = combined
            best_name = name
            best_pipeline = pipe

    if verbose:
        print(f"\nBest model: {best_name}  (combined score={best_score:.4f})")

    # 3. Extract feature importance / coefficients
    classifier = best_pipeline.named_steps["classifier"]
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names_out(fitted_preprocessor, num_cols, cat_cols)

    importance_pairs = []
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        importance_pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1], reverse=True
        )
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        importance_pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1], reverse=True
        )

    top_features = [{"feature": f, "importance": round(v, 6)} for f, v in importance_pairs[:20]]

    # 4. Save artifacts
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    joblib.dump({"preprocessor": preprocessor, "num_cols": num_cols, "cat_cols": cat_cols},
                PREPROCESSOR_PATH)

    # 5. Save metadata
    metadata = {
        "best_model": best_name,
        "combined_score": round(best_score, 4),
        "metrics": results,
        "top_features": top_features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_names": feature_names,
        "class_labels": {0: "No", 1: "Yes"},
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Artifacts saved to {MODEL_DIR}")

    return metadata


if __name__ == "__main__":
    train_and_select()

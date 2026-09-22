"""
explainability.py
Advanced explainable AI for AttritionIQ predictions.
Uses SHAP for Logistic Regression and coefficient/importance-based
explanations for other model types.

Disclaimer: These factors are associated with the model prediction
and do not establish causation.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")

LABEL_MAP = {
    "OverTime": "Overtime",
    "JobSatisfaction": "Job Satisfaction",
    "WorkLifeBalance": "Work-Life Balance",
    "MonthlyIncome": "Monthly Income",
    "YearsAtCompany": "Years at Company",
    "DistanceFromHome": "Distance From Home",
    "StockOptionLevel": "Stock Option Level",
    "JobLevel": "Job Level",
    "TotalWorkingYears": "Total Working Years",
    "YearsInCurrentRole": "Years in Current Role",
    "YearsSinceLastPromotion": "Years Since Last Promotion",
    "YearsWithCurrManager": "Years With Current Manager",
    "NumCompaniesWorked": "No. of Companies Worked",
    "TrainingTimesLastYear": "Training Times (Last Year)",
    "PercentSalaryHike": "Salary Hike %",
    "BusinessTravel": "Business Travel",
    "EnvironmentSatisfaction": "Environment Satisfaction",
    "RelationshipSatisfaction": "Relationship Satisfaction",
    "JobInvolvement": "Job Involvement",
    "Age": "Age",
    "Department": "Department",
    "JobRole": "Job Role",
    "EducationField": "Education Field",
    "Education": "Education Level",
    "PerformanceRating": "Performance Rating",
    "Gender": "Gender",
    "MaritalStatus": "Marital Status",
    "DailyRate": "Daily Rate",
    "HourlyRate": "Hourly Rate",
    "MonthlyRate": "Monthly Rate",
}


def _humanize_feature(name: str) -> str:
    """Convert internal OHE or raw feature name to readable label."""
    if "_" in name:
        parts = name.split("_", 1)
        col = parts[0]
        val = parts[1]
        readable = LABEL_MAP.get(col, col)
        return f"{readable}: {val}"
    return LABEL_MAP.get(name, name)


def explain_prediction_shap(emp_df: pd.DataFrame, pipeline, top_n: int = 5) -> list:
    """
    Use SHAP LinearExplainer for Logistic Regression.
    Returns list of dicts with feature, value, shap_value, direction.
    """
    try:
        import shap
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        X_transformed = preprocessor.transform(emp_df)

        # Build feature names
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        num_cols = meta["num_cols"]
        cat_cols = meta["cat_cols"]
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        all_feature_names = num_cols + cat_feature_names

        # SHAP LinearExplainer works well with logistic regression
        explainer = shap.LinearExplainer(
            classifier, X_transformed, feature_perturbation="correlation_dependent"
        )
        shap_values = explainer.shap_values(X_transformed)

        # shap_values shape: (1, n_features)
        sv = shap_values[0] if shap_values.ndim == 2 else shap_values

        ranked = sorted(
            zip(all_feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        factors = []
        seen_base = set()
        for fname, sv_val in ranked:
            base = fname.split("_")[0] if "_" in fname else fname
            if base not in seen_base:
                seen_base.add(base)
                direction = "risk-increasing" if sv_val > 0 else "protective"
                factors.append({
                    "feature": _humanize_feature(fname),
                    "raw_feature": fname,
                    "shap_value": round(float(sv_val), 4),
                    "direction": direction,
                    "contribution": "positive" if sv_val > 0 else "negative",
                })
            if len(factors) >= top_n:
                break
        return factors

    except Exception:
        return []


def explain_prediction_coef(emp_df: pd.DataFrame, pipeline, meta: dict, top_n: int = 5) -> list:
    """
    Coefficient × transformed-value explanation for Logistic Regression.
    Falls back to feature_importances_ for tree models.
    """
    num_cols = meta.get("num_cols", [])
    cat_cols = meta.get("cat_cols", [])

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(emp_df)

    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    if hasattr(classifier, "coef_"):
        coefs = classifier.coef_[0]
        signed_scores = coefs * X_transformed[0]   # signed contribution
        abs_scores = np.abs(signed_scores)
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        abs_scores = importances * np.abs(X_transformed[0])
        signed_scores = abs_scores  # no sign available
    else:
        return [{"feature": f, "direction": "unknown", "contribution": "positive"} for f in
                [_humanize_feature(n) for n in all_feature_names[:top_n]]]

    ranked = sorted(
        zip(all_feature_names, signed_scores, abs_scores),
        key=lambda x: x[2],
        reverse=True,
    )

    factors = []
    seen_base = set()
    for fname, signed_val, abs_val in ranked:
        base = fname.split("_")[0] if "_" in fname else fname
        if base not in seen_base:
            seen_base.add(base)
            direction = "risk-increasing" if signed_val > 0 else "protective"
            factors.append({
                "feature": _humanize_feature(fname),
                "raw_feature": fname,
                "shap_value": round(float(signed_val), 4),
                "direction": direction,
                "contribution": "positive" if signed_val > 0 else "negative",
            })
        if len(factors) >= top_n:
            break
    return factors


def get_explanation(emp_df: pd.DataFrame, pipeline, meta: dict, top_n: int = 5) -> dict:
    """
    Main explainability entry point.
    Tries SHAP first, falls back to coefficient/importance approach.
    Returns structured explanation dict.
    """
    model_name = meta.get("best_model", "")

    # Try SHAP for Logistic Regression
    factors = []
    method = "coefficient"
    if "Logistic" in model_name:
        factors = explain_prediction_shap(emp_df, pipeline, top_n=top_n)
        if factors:
            method = "shap"

    if not factors:
        factors = explain_prediction_coef(emp_df, pipeline, meta, top_n=top_n)
        method = "coefficient"

    risk_factors = [f for f in factors if f["contribution"] == "positive"]
    protective_factors = [f for f in factors if f["contribution"] == "negative"]

    return {
        "factors": factors,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "method": method,
        "disclaimer": (
            "These factors are associated with the model prediction and "
            "do not establish causation."
        ),
    }

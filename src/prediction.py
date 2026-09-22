"""
prediction.py
Single-employee prediction: preprocessing, model inference, risk level,
feature attribution and HR recommendations.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import sys

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")

# ─── Thresholds ───────────────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    "Low": (0.0, 0.30),
    "Medium": (0.30, 0.60),
    "High": (0.60, 1.01),
}

# Actionable features for What-If scenario analysis
SCENARIO_FEATURES = {
    "OverTime": {
        "type": "categorical",
        "options": ["No", "Yes"],
        "label": "Overtime",
    },
    "JobSatisfaction": {
        "type": "numeric",
        "min": 1, "max": 4, "step": 1,
        "label": "Job Satisfaction (1–4)",
    },
    "WorkLifeBalance": {
        "type": "numeric",
        "min": 1, "max": 4, "step": 1,
        "label": "Work-Life Balance (1–4)",
    },
    "JobInvolvement": {
        "type": "numeric",
        "min": 1, "max": 4, "step": 1,
        "label": "Job Involvement (1–4)",
    },
    "YearsAtCompany": {
        "type": "numeric",
        "min": 0, "max": 40, "step": 1,
        "label": "Years at Company",
    },
    "YearsSinceLastPromotion": {
        "type": "numeric",
        "min": 0, "max": 20, "step": 1,
        "label": "Years Since Last Promotion",
    },
    "EnvironmentSatisfaction": {
        "type": "numeric",
        "min": 1, "max": 4, "step": 1,
        "label": "Environment Satisfaction (1–4)",
    },
    "RelationshipSatisfaction": {
        "type": "numeric",
        "min": 1, "max": 4, "step": 1,
        "label": "Relationship Satisfaction (1–4)",
    },
    "TrainingTimesLastYear": {
        "type": "numeric",
        "min": 0, "max": 10, "step": 1,
        "label": "Training Times Last Year",
    },
    "StockOptionLevel": {
        "type": "numeric",
        "min": 0, "max": 3, "step": 1,
        "label": "Stock Option Level (0–3)",
    },
}


def classify_risk(prob: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= prob < hi:
            return level
    return "High"


def classify_risk_at_threshold(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "High" if prob >= 0.60 else "Medium"
    return "Low" if prob < 0.30 else "Medium"


# ─── Loader ───────────────────────────────────────────────────────────────────

_pipeline = None
_metadata = None


def load_artifacts():
    global _pipeline, _metadata
    if _pipeline is None:
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                "best_model.pkl not found. Run src/train_model.py first."
            )
        _pipeline = joblib.load(BEST_MODEL_PATH)
    if _metadata is None:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _pipeline, _metadata


# ─── Feature attribution (lightweight) ───────────────────────────────────────

def get_top_factors(employee_df: pd.DataFrame, pipeline, metadata: dict, top_n: int = 5) -> list:
    """
    Approximate per-prediction factor attribution.
    For tree-based models we use global feature importances weighted by
    the deviation of the employee value from the training mean.
    This is a lightweight alternative to SHAP.
    """
    top_features = metadata.get("top_features", [])
    feature_names = metadata.get("feature_names", [])
    num_cols = metadata.get("num_cols", [])
    cat_cols = metadata.get("cat_cols", [])

    # Get the transformed feature vector
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(employee_df)

    # Map transformed features to importances
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + cat_feature_names

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        # Fall back to global top features
        return [f["feature"] for f in top_features[:top_n]]

    # Score = importance * |transformed value|
    scores = importances * np.abs(X_transformed[0])
    ranked = sorted(zip(all_feature_names, scores), key=lambda x: x[1], reverse=True)

    factors = []
    seen_base = set()
    for fname, score in ranked:
        # Humanize the feature name
        base = fname.split("_")[0] if "_" in fname else fname
        if base not in seen_base:
            seen_base.add(base)
            factors.append(_humanize(fname, employee_df))
        if len(factors) >= top_n:
            break
    return factors


def _humanize(feature_name: str, emp_df: pd.DataFrame) -> str:
    """Convert internal feature names to readable HR labels."""
    # Categorical OHE: "OverTime_Yes" → "Overtime: Yes"
    parts = feature_name.split("_", 1)
    col = parts[0]
    val = parts[1] if len(parts) > 1 else None

    label_map = {
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
    }

    readable_col = label_map.get(col, col)

    if val is not None:
        # Try to get the actual employee value
        if col in emp_df.columns:
            emp_val = str(emp_df[col].iloc[0])
            return f"{readable_col}: {emp_val}"
        return f"{readable_col}: {val}"
    else:
        if col in emp_df.columns:
            emp_val = emp_df[col].iloc[0]
            return f"{readable_col} = {emp_val}"
        return readable_col


# ─── Recommendations engine ───────────────────────────────────────────────────

def generate_recommendations(employee: dict, factors: list) -> list:
    """
    Rule-based HR recommendation engine.
    Recommendations are decision-support suggestions, NOT automated HR decisions.
    Protected attributes (Gender, MaritalStatus) are NOT used.
    """
    recs = []
    added = set()

    def add(msg):
        if msg not in added:
            recs.append(msg)
            added.add(msg)

    # Overtime
    if str(employee.get("OverTime", "")).strip().lower() in ["yes", "1", "true"]:
        add("Consider reviewing workload distribution and overtime requirements "
            "to support sustainable work patterns.")

    # Job Satisfaction (1=Low, 4=Very High)
    try:
        js = int(employee.get("JobSatisfaction", 3))
        if js <= 2:
            add("Consider scheduling a manager check-in or employee engagement "
                "discussion to understand satisfaction concerns.")
    except (ValueError, TypeError):
        pass

    # Work-Life Balance (1=Bad, 4=Best)
    try:
        wlb = int(employee.get("WorkLifeBalance", 3))
        if wlb <= 2:
            add("Consider exploring workload flexibility, schedule adjustments, "
                "or remote work options to improve work-life balance.")
    except (ValueError, TypeError):
        pass

    # Environment Satisfaction
    try:
        es = int(employee.get("EnvironmentSatisfaction", 3))
        if es <= 2:
            add("Consider reviewing the employee's working environment and team "
                "dynamics as part of an engagement discussion.")
    except (ValueError, TypeError):
        pass

    # Years since last promotion
    try:
        yslp = int(employee.get("YearsSinceLastPromotion", 0))
        if yslp >= 4:
            add("Consider reviewing career development pathways, mentoring "
                "opportunities, or progression timelines for this employee.")
    except (ValueError, TypeError):
        pass

    # Monthly income vs. job level
    try:
        income = float(employee.get("MonthlyIncome", 5000))
        job_level = int(employee.get("JobLevel", 2))
        expected_floor = job_level * 1500
        if income < expected_floor:
            add("Review compensation competitiveness relative to the employee's "
                "job level and market benchmarks.")
    except (ValueError, TypeError):
        pass

    # Distance from home
    try:
        dist = int(employee.get("DistanceFromHome", 5))
        if dist >= 20:
            add("Consider whether remote or hybrid work arrangements could reduce "
                "commute burden for this employee.")
    except (ValueError, TypeError):
        pass

    # Stock option level
    try:
        sol = int(employee.get("StockOptionLevel", 0))
        if sol == 0:
            add("Consider whether the employee could benefit from long-term "
                "incentive programs such as stock options or retention bonuses.")
    except (ValueError, TypeError):
        pass

    # Training
    try:
        training = int(employee.get("TrainingTimesLastYear", 2))
        if training <= 1:
            add("Consider increasing access to training, upskilling, or "
                "professional development programmes.")
    except (ValueError, TypeError):
        pass

    # Relationship satisfaction
    try:
        rs = int(employee.get("RelationshipSatisfaction", 3))
        if rs <= 2:
            add("Consider facilitating team-building activities or conflict "
                "resolution support to improve interpersonal relationships.")
    except (ValueError, TypeError):
        pass

    if not recs:
        add("Maintain regular engagement check-ins and monitor key satisfaction "
            "indicators over time.")

    return recs


# ─── Build employee DataFrame ─────────────────────────────────────────────────

def build_employee_df(employee_data: dict, metadata: dict) -> pd.DataFrame:
    """Build a single-row DataFrame from raw employee_data dict."""
    num_cols = metadata["num_cols"]
    cat_cols = metadata["cat_cols"]
    all_cols = num_cols + cat_cols

    row = {}
    for col in all_cols:
        val = employee_data.get(col, None)
        if val is None or val == "":
            if col in num_cols:
                row[col] = 0
            else:
                row[col] = "Unknown"
        else:
            if col in num_cols:
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = 0.0
            else:
                row[col] = str(val)
    return pd.DataFrame([row])


# ─── Main prediction function ─────────────────────────────────────────────────

def predict_employee(employee_data: dict) -> dict:
    """
    Accepts a dict of employee fields.
    Returns prediction, probability, risk_level, top_factors,
    explanation, recommendations.
    """
    from explainability import get_explanation

    pipeline, metadata = load_artifacts()
    emp_df = build_employee_df(employee_data, metadata)

    # Predict
    probability = float(pipeline.predict_proba(emp_df)[0][1])
    prediction_int = int(pipeline.predict(emp_df)[0])
    prediction_label = "Yes" if prediction_int == 1 else "No"
    risk_level = classify_risk(probability)

    # Top factors (legacy simple list for backward compatibility)
    top_factors = get_top_factors(emp_df, pipeline, metadata, top_n=5)

    # Advanced explanation with direction
    explanation = get_explanation(emp_df, pipeline, metadata, top_n=5)

    # Recommendations
    recommendations = generate_recommendations(employee_data, top_factors)

    return {
        "prediction": prediction_label,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "top_factors": top_factors,
        "explanation": explanation,
        "recommendations": recommendations,
    }


# ─── Scenario analysis ────────────────────────────────────────────────────────

def scenario_predict(base_employee_data: dict, overrides: dict) -> dict:
    """
    What-If scenario analysis.
    Applies overrides to base_employee_data and re-runs prediction.

    This is a MODEL SIMULATION of hypothetical changes.
    It does NOT predict whether an employee will actually leave.
    """
    pipeline, metadata = load_artifacts()

    # Build modified employee data
    scenario_data = dict(base_employee_data)
    scenario_data.update(overrides)

    # Base prediction
    base_df = build_employee_df(base_employee_data, metadata)
    base_prob = float(pipeline.predict_proba(base_df)[0][1])
    base_risk = classify_risk(base_prob)

    # Scenario prediction
    scenario_df = build_employee_df(scenario_data, metadata)
    scenario_prob = float(pipeline.predict_proba(scenario_df)[0][1])
    scenario_risk = classify_risk(scenario_prob)

    diff_pp = round((scenario_prob - base_prob) * 100, 1)

    return {
        "current_probability": round(base_prob, 4),
        "scenario_probability": round(scenario_prob, 4),
        "current_risk": base_risk,
        "scenario_risk": scenario_risk,
        "difference_pp": diff_pp,
        "overrides": overrides,
        "disclaimer": (
            "This is a model simulation of hypothetical changes. "
            "It does not predict whether an employee will actually leave. "
            "Changing a feature value in the model does not establish that "
            "the change will prevent attrition."
        ),
    }


# ─── Threshold analysis ───────────────────────────────────────────────────────

def threshold_analysis(employee_data: dict = None) -> dict:
    """
    Compute per-threshold metrics on the test set.
    If employee_data is provided, also compute per-threshold prediction for that employee.
    """
    from data_preprocessing import load_dataset, prepare_data
    from sklearn.metrics import precision_score, recall_score, f1_score

    pipeline, metadata = load_artifacts()
    df = load_dataset()
    _, X_test, _, y_test = prepare_data(df)[:4]

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        n_high = int(y_pred_t.sum())
        p = round(float(precision_score(y_test, y_pred_t, zero_division=0)) * 100, 1)
        r = round(float(recall_score(y_test, y_pred_t, zero_division=0)) * 100, 1)
        f = round(float(f1_score(y_test, y_pred_t, zero_division=0)) * 100, 1)
        results.append({
            "threshold": t,
            "precision": p,
            "recall": r,
            "f1": f,
            "n_high_risk": n_high,
            "is_default": t == 0.50,
        })

    # Per-employee threshold analysis
    employee_thresholds = []
    if employee_data:
        emp_df = build_employee_df(employee_data, metadata)
        emp_prob = float(pipeline.predict_proba(emp_df)[0][1])
        for t in thresholds:
            pred = "Attrition" if emp_prob >= t else "No Attrition"
            risk = "High" if emp_prob >= 0.60 else ("Medium" if emp_prob >= 0.30 else "Low")
            employee_thresholds.append({
                "threshold": t,
                "prediction": pred,
                "risk": risk,
                "probability": round(emp_prob, 4),
            })

    return {
        "thresholds": results,
        "employee_thresholds": employee_thresholds,
        "note": (
            "The dataset is class-imbalanced (~16% attrition). "
            "Lowering the threshold increases recall (fewer missed at-risk employees) "
            "but reduces precision (more false positives). "
            "The default threshold of 0.50 is used unless explicitly changed."
        ),
    }

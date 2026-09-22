"""
data_preprocessing.py
Loads, cleans and prepares the IBM HR Attrition dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import json
import os

# ─── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Columns known to carry zero variance or be irrelevant identifiers.
# These are verified against the actual dataset during load.
CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]
ID_COLUMNS = ["EmployeeNumber"]

TARGET_COLUMN = "Attrition"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV and return raw DataFrame."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Dataset not found at {abs_path}. "
            "Please place WA_Fn-UseC_-HR-Employee-Attrition.csv in the data/ directory."
        )
    df = pd.read_csv(abs_path)
    return df


# ─── EDA helpers ──────────────────────────────────────────────────────────────

def dataset_overview(df: pd.DataFrame) -> dict:
    """Return a structured overview dict for EDA / API consumption."""
    overview = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "target_rate": round(
            (df[TARGET_COLUMN] == "Yes").mean() * 100, 2
        ),
        "unique_values": {col: int(df[col].nunique()) for col in df.columns},
    }
    return overview


def compute_attrition_stats(df: pd.DataFrame) -> dict:
    """Compute grouped attrition statistics used in dashboard and insights."""
    stats = {}
    df = df.copy()

    def rate_by(group_col, observed=False):
        """Compute attrition rate per group, avoiding FutureWarning."""
        binary = (df[TARGET_COLUMN] == "Yes").astype(float)
        tmp = df[[group_col]].copy()
        tmp["_val"] = binary
        result = (
            tmp.groupby(group_col, observed=observed)["_val"]
            .mean()
            .mul(100)
            .round(1)
            .reset_index()
            .rename(columns={"_val": "attrition_rate"})
        )
        return result

    # Department
    stats["by_department"] = rate_by("Department").to_dict(orient="records")

    # Job Role
    stats["by_job_role"] = rate_by("JobRole").to_dict(orient="records")

    # Overtime
    stats["by_overtime"] = rate_by("OverTime").to_dict(orient="records")

    # Job Satisfaction
    stats["by_job_satisfaction"] = rate_by("JobSatisfaction").to_dict(orient="records")

    # Age group
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[17, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"]
    )
    stats["by_age_group"] = rate_by("AgeGroup", observed=True).to_dict(orient="records")

    # Years at company buckets
    df["TenureBucket"] = pd.cut(
        df["YearsAtCompany"], bins=[-1, 2, 5, 10, 20, 100],
        labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "11-20 yrs", "20+ yrs"]
    )
    stats["by_tenure"] = rate_by("TenureBucket", observed=True).to_dict(orient="records")

    # Business Travel
    stats["by_business_travel"] = rate_by("BusinessTravel").to_dict(orient="records")

    # KPIs
    stats["kpi"] = {
        "total_employees": int(len(df)),
        "employees_left": int((df[TARGET_COLUMN] == "Yes").sum()),
        "employees_retained": int((df[TARGET_COLUMN] == "No").sum()),
        "attrition_rate": round((df[TARGET_COLUMN] == "Yes").mean() * 100, 1),
        "avg_monthly_income": round(df["MonthlyIncome"].mean(), 0),
        "avg_years_at_company": round(df["YearsAtCompany"].mean(), 1),
    }

    return stats


# ─── Preprocessing pipeline ───────────────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame):
    """
    Infer feature types from the DataFrame and build a ColumnTransformer
    preprocessing pipeline. Returns (preprocessor, feature_names_in, cat_cols, num_cols).
    """
    # Drop columns that are always removed
    drop_cols = CONSTANT_COLUMNS + ID_COLUMNS + [TARGET_COLUMN]
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    # Identify numeric vs categorical columns
    num_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    return preprocessor, num_cols, cat_cols


def prepare_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test, preprocessor, feature_names, num_cols, cat_cols
    """
    # Encode target
    df = df.copy()
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)

    drop_cols = CONSTANT_COLUMNS + ID_COLUMNS + [TARGET_COLUMN]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN]

    preprocessor, num_cols, cat_cols = build_preprocessor(df.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in df.columns else df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols


def get_feature_names_out(preprocessor, num_cols, cat_cols):
    """Extract feature names after fitting the preprocessor."""
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    return num_cols + cat_feature_names

"""
data_quality.py
Data quality monitoring for AttritionIQ.

Calculates data quality metrics dynamically from the actual dataset.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import (
    load_dataset, CONSTANT_COLUMNS, ID_COLUMNS, TARGET_COLUMN
)


def compute_data_quality(df: pd.DataFrame = None) -> dict:
    """
    Compute comprehensive data quality metrics from the actual dataset.
    Returns a structured dict with indicators and status flags.
    """
    if df is None:
        df = load_dataset()

    n_rows, n_cols = df.shape

    # Missing values
    missing = df.isnull().sum()
    missing_dict = {col: int(v) for col, v in missing.items() if v > 0}
    total_missing = int(missing.sum())

    # Duplicate rows
    n_duplicates = int(df.duplicated().sum())

    # Target distribution
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    n_yes = int((df[TARGET_COLUMN] == "Yes").sum())
    n_no = int((df[TARGET_COLUMN] == "No").sum())
    majority = max(n_yes, n_no)
    minority = min(n_yes, n_no)
    imbalance_ratio = round(majority / minority, 2) if minority > 0 else float("inf")
    class_imbalance_pct = round(minority / n_rows * 100, 1)

    # Constant columns that were removed
    constant_cols_present = [c for c in CONSTANT_COLUMNS if c in df.columns]
    constant_col_details = {}
    for c in constant_cols_present:
        unique_vals = df[c].unique().tolist()
        constant_col_details[c] = unique_vals[0] if len(unique_vals) == 1 else unique_vals

    # ID columns excluded
    id_cols_present = [c for c in ID_COLUMNS if c in df.columns]

    # Column type breakdown
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Outlier detection (IQR) for numeric columns
    outlier_summary = {}
    numeric_feature_cols = [c for c in numeric_cols
                            if c not in CONSTANT_COLUMNS + ID_COLUMNS + [TARGET_COLUMN]]
    for col in numeric_feature_cols[:10]:  # top 10 for display
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_out > 0:
            outlier_summary[col] = {
                "count": n_out,
                "pct": round(n_out / n_rows * 100, 1)
            }

    # Column-level missing detail (all columns)
    missing_by_col = []
    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        missing_by_col.append({
            "column": col,
            "missing": n_miss,
            "missing_pct": round(n_miss / n_rows * 100, 1),
            "status": "ok" if n_miss == 0 else "warning"
        })

    # Status indicators
    status = {
        "missing_values": "healthy" if total_missing == 0 else "review",
        "duplicates": "healthy" if n_duplicates == 0 else "review",
        "class_balance": "healthy" if imbalance_ratio <= 3.0 else "review",
        "dataset_size": "healthy" if n_rows >= 1000 else "review",
    }

    # Summary checks
    checks = [
        {
            "label": "Missing Values",
            "value": f"{total_missing} missing values",
            "status": status["missing_values"],
            "detail": "No missing values detected." if total_missing == 0
                      else f"Columns with missing data: {list(missing_dict.keys())}",
        },
        {
            "label": "Duplicate Rows",
            "value": f"{n_duplicates} duplicates",
            "status": status["duplicates"],
            "detail": "No duplicate rows found." if n_duplicates == 0
                      else f"{n_duplicates} duplicate rows detected.",
        },
        {
            "label": "Class Balance",
            "value": f"{n_yes} Yes / {n_no} No",
            "status": status["class_balance"],
            "detail": (f"Minority class ({minority}) is {class_imbalance_pct}% of dataset. "
                       f"Imbalance ratio: {imbalance_ratio}:1. "
                       "Class weights are applied during model training."),
        },
        {
            "label": "Dataset Size",
            "value": f"{n_rows:,} rows × {n_cols} cols",
            "status": status["dataset_size"],
            "detail": "Dataset size is sufficient for modelling."
                      if n_rows >= 1000 else "Dataset may be too small.",
        },
    ]

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_missing": total_missing,
        "missing_by_col": missing_by_col,
        "n_duplicates": n_duplicates,
        "target_distribution": {
            "Yes": n_yes,
            "No": n_no,
            "attrition_rate_pct": round(n_yes / n_rows * 100, 1),
        },
        "class_imbalance_ratio": imbalance_ratio,
        "class_imbalance_pct": class_imbalance_pct,
        "constant_cols_removed": constant_col_details,
        "id_cols_excluded": id_cols_present,
        "numeric_cols_count": len(numeric_cols),
        "categorical_cols_count": len(categorical_cols),
        "outliers": outlier_summary,
        "checks": checks,
        "status": status,
    }

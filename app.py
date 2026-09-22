"""
app.py – AttritionIQ Flask application entry point.
"""

import os
import sys
import json
import traceback

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_dataset, dataset_overview, compute_attrition_stats
from evaluate_model import (
    load_metadata, get_model_comparison,
    get_roc_curve_data, get_confusion_matrix_data, get_feature_importance
)
from prediction import predict_employee, scenario_predict, threshold_analysis, SCENARIO_FEATURES
from data_quality import compute_data_quality
from fairness import compute_fairness_metrics

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ─── Dataset cache ────────────────────────────────────────────────────────────

_df_cache = None
_stats_cache = None


def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = load_dataset()
    return _df_cache


def get_stats():
    global _stats_cache
    if _stats_cache is None:
        df = get_df()
        _stats_cache = compute_attrition_stats(df)
    return _stats_cache


# ─── Page routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    try:
        stats = get_stats()
        return render_template("index.html", kpi=stats["kpi"])
    except FileNotFoundError as e:
        return render_template("error.html", message=str(e)), 500


@app.route("/dashboard")
def dashboard():
    try:
        stats = get_stats()
        return render_template("dashboard.html", stats=stats, kpi=stats["kpi"])
    except FileNotFoundError as e:
        return render_template("error.html", message=str(e)), 500


@app.route("/prediction")
def prediction_page():
    try:
        df = get_df()
        # Pass column options to the form
        options = {
            "departments": sorted(df["Department"].unique().tolist()),
            "job_roles": sorted(df["JobRole"].unique().tolist()),
            "education_fields": sorted(df["EducationField"].unique().tolist()),
            "business_travel": sorted(df["BusinessTravel"].unique().tolist()),
        }
        return render_template("prediction.html", options=options,
                               scenario_features=SCENARIO_FEATURES)
    except FileNotFoundError as e:
        return render_template("error.html", message=str(e)), 500


@app.route("/insights")
def insights():
    try:
        df = get_df()
        stats = get_stats()
        overview = dataset_overview(df)
        return render_template("insights.html", stats=stats, overview=overview)
    except FileNotFoundError as e:
        return render_template("error.html", message=str(e)), 500


@app.route("/fairness")
def fairness_page():
    """Fairness / Bias Audit page."""
    return render_template("fairness.html")


@app.route("/data-quality")
def data_quality_page():
    """Data Quality Monitor page."""
    try:
        df = get_df()
        quality = compute_data_quality(df)
        return render_template("data_quality.html", quality=quality)
    except FileNotFoundError as e:
        return render_template("error.html", message=str(e)), 500


@app.route("/executive")
def executive_page():
    """Executive HR Summary page."""
    try:
        df = get_df()
        stats = get_stats()
        return render_template("executive.html", stats=stats, kpi=stats["kpi"])
    except FileNotFoundError as e:
        return render_template("error.html", message=str(e)), 500


@app.route("/about")
def about():
    return render_template("about.html")


# ─── API routes ───────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400
        result = predict_employee(data)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    """What-If scenario analysis endpoint."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400
        base_data = data.get("employee", {})
        overrides = data.get("overrides", {})
        if not base_data or not overrides:
            return jsonify({"error": "Provide 'employee' and 'overrides' keys."}), 400
        result = scenario_predict(base_data, overrides)
        return jsonify(result)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Scenario analysis failed: {str(e)}"}), 500


@app.route("/api/threshold-analysis", methods=["POST"])
def api_threshold_analysis():
    """Threshold analysis endpoint."""
    try:
        data = request.get_json(force=True) or {}
        employee_data = data.get("employee", None)
        result = threshold_analysis(employee_data)
        return jsonify(result)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Threshold analysis failed: {str(e)}"}), 500


@app.route("/api/fairness")
def api_fairness():
    """Fairness audit data endpoint."""
    try:
        df = get_df()
        result = compute_fairness_metrics(df)
        return jsonify(result)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-quality")
def api_data_quality():
    """Data quality metrics endpoint."""
    try:
        df = get_df()
        result = compute_data_quality(df)
        return jsonify(result)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/executive-summary")
def api_executive_summary():
    """Executive HR summary data endpoint."""
    try:
        df = get_df()
        stats = get_stats()
        meta = load_metadata()

        # Predicted risk counts – run model on full dataset
        import joblib
        import numpy as np
        from data_preprocessing import CONSTANT_COLUMNS, ID_COLUMNS
        pipeline = joblib.load(
            os.path.join(os.path.dirname(__file__), "model", "best_model.pkl")
        )
        drop_cols = [c for c in CONSTANT_COLUMNS + ID_COLUMNS + ["Attrition"]
                     if c in df.columns]
        X_all = df.drop(columns=drop_cols)
        probs = pipeline.predict_proba(X_all)[:, 1]
        from prediction import classify_risk
        risks = [classify_risk(p) for p in probs]
        n_high = int(risks.count("High"))
        n_medium = int(risks.count("Medium"))
        n_low = int(risks.count("Low"))

        return jsonify({
            "kpi": stats["kpi"],
            "by_department": stats["by_department"],
            "by_job_role": stats["by_job_role"],
            "by_overtime": stats["by_overtime"],
            "by_job_satisfaction": stats["by_job_satisfaction"],
            "predicted_risk": {
                "high": n_high,
                "medium": n_medium,
                "low": n_low,
            },
            "best_model": meta["best_model"],
        })
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-monitoring")
def api_model_monitoring():
    """Model monitoring data endpoint."""
    try:
        meta = load_metadata()
        from data_preprocessing import load_dataset, prepare_data
        df = load_dataset()
        X_train, X_test, y_train, y_test, _, _, _ = prepare_data(df)

        # Training date from model file mtime
        model_path = os.path.join(os.path.dirname(__file__), "model", "best_model.pkl")
        import time
        mtime = os.path.getmtime(model_path)
        from datetime import datetime
        training_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        best = meta["best_model"]
        m = meta["metrics"][best]

        return jsonify({
            "best_model": best,
            "metrics": {
                "accuracy": round(m["accuracy"] * 100, 1),
                "precision": round(m["precision"] * 100, 1),
                "recall": round(m["recall"] * 100, 1),
                "f1": round(m["f1"] * 100, 1),
                "roc_auc": round(m["roc_auc"] * 100, 1),
            },
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_date": training_date,
            "selection_criterion": "Equal-weight combination of F1 + Recall + ROC-AUC",
            "combined_score": meta.get("combined_score"),
        })
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics")
def api_metrics():
    try:
        comparison = get_model_comparison()
        meta = load_metadata()
        roc = get_roc_curve_data()
        cm = get_confusion_matrix_data()
        return jsonify({
            "best_model": meta["best_model"],
            "model_comparison": comparison,
            "roc_curve": roc,
            "confusion_matrix": cm,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/feature-importance")
def api_feature_importance():
    try:
        features = get_feature_importance()
        return jsonify({"feature_importance": features})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard-data")
def api_dashboard_data():
    """Filtered dashboard data endpoint."""
    try:
        import pandas as pd
        df = get_df().copy()

        # Apply filters
        dept = request.args.get("department")
        role = request.args.get("job_role")
        gender = request.args.get("gender")
        travel = request.args.get("business_travel")
        overtime = request.args.get("overtime")
        job_level = request.args.get("job_level")

        if dept and dept != "All":
            df = df[df["Department"] == dept]
        if role and role != "All":
            df = df[df["JobRole"] == role]
        if gender and gender != "All":
            df = df[df["Gender"] == gender]
        if travel and travel != "All":
            df = df[df["BusinessTravel"] == travel]
        if overtime and overtime != "All":
            df = df[df["OverTime"] == overtime]
        if job_level and job_level != "All":
            try:
                df = df[df["JobLevel"] == int(job_level)]
            except ValueError:
                pass

        if df.empty:
            return jsonify({"error": "No data matches the selected filters."}), 404

        stats = compute_attrition_stats(df)
        return jsonify(stats)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/filter-options")
def api_filter_options():
    """Return unique values for filter dropdowns."""
    try:
        df = get_df()
        options = {
            "departments": sorted(df["Department"].unique().tolist()),
            "job_roles": sorted(df["JobRole"].unique().tolist()),
            "genders": sorted(df["Gender"].unique().tolist()),
            "business_travel": sorted(df["BusinessTravel"].unique().tolist()),
            "overtime": sorted(df["OverTime"].unique().tolist()),
            "job_levels": sorted(df["JobLevel"].unique().tolist()),
        }
        return jsonify(options)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download-report", methods=["POST"])
def api_download_report():
    """Generate and download a PDF prediction report."""
    try:
        import io as _io
        from pdf_report import get_report_mimetype

        data = request.get_json(force=True) or {}
        prediction_result = data.get("prediction_result", {})
        employee_data = data.get("employee_data", None)
        scenario_result = data.get("scenario_result", None)

        if not prediction_result:
            return jsonify({"error": "prediction_result is required"}), 400

        report_bytes, mimetype, filename = get_report_mimetype(
            prediction_result, employee_data, scenario_result
        )
        return send_file(
            _io.BytesIO(report_bytes),
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500


# ─── Error handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found (404)."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", message="Internal server error (500)."), 500


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AttritionIQ – IBM Employee Attrition Analytics")
    print("=" * 60)
    print("  Open: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)

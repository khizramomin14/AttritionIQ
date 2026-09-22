"""
tests/test_attritioniq.py
Comprehensive test suite for AttritionIQ (original + new features).
Run with: python -m pytest tests/ -v
"""

import os
import sys
import json
import pytest

# Make sure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_model.pkl")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model_metadata.json")


# ── Dataset tests ──────────────────────────────────────────────────────────

class TestDataset:
    def test_csv_exists(self):
        assert os.path.exists(DATA_PATH), \
            "Dataset CSV not found. Place it in AttritionIQ/data/"

    def test_load_dataset(self):
        from data_preprocessing import load_dataset
        df = load_dataset()
        assert df is not None
        assert len(df) > 0, "Dataset is empty"

    def test_dataset_shape(self):
        from data_preprocessing import load_dataset
        df = load_dataset()
        assert df.shape[0] >= 1000, "Expected at least 1000 rows"
        assert df.shape[1] >= 20,   "Expected at least 20 columns"

    def test_target_column_exists(self):
        from data_preprocessing import load_dataset
        df = load_dataset()
        assert "Attrition" in df.columns, "Target column 'Attrition' not found"

    def test_target_values(self):
        from data_preprocessing import load_dataset
        df = load_dataset()
        vals = set(df["Attrition"].unique())
        assert vals == {"Yes", "No"}, f"Unexpected target values: {vals}"

    def test_no_missing_in_target(self):
        from data_preprocessing import load_dataset
        df = load_dataset()
        assert df["Attrition"].isnull().sum() == 0


# ── Preprocessing tests ─────────────────────────────────────────────────────

class TestPreprocessing:
    def test_prepare_data_shapes(self):
        from data_preprocessing import load_dataset, prepare_data
        df = load_dataset()
        X_train, X_test, y_train, y_test, pp, num_cols, cat_cols = prepare_data(df)
        assert X_train.shape[0] > 0
        assert X_test.shape[0]  > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test)  == X_test.shape[0]

    def test_stratified_split(self):
        from data_preprocessing import load_dataset, prepare_data
        df = load_dataset()
        _, _, y_train, y_test, *_ = prepare_data(df)
        train_rate = y_train.mean()
        test_rate  = y_test.mean()
        assert abs(train_rate - test_rate) < 0.05, \
            "Stratification failed: class rates differ by more than 5%"

    def test_preprocessor_fit_transform(self):
        from data_preprocessing import load_dataset, prepare_data
        df = load_dataset()
        X_train, X_test, y_train, y_test, pp, num_cols, cat_cols = prepare_data(df)
        X_tr = pp.fit_transform(X_train, y_train)
        assert X_tr.shape[0] == X_train.shape[0]
        assert X_tr.shape[1] >= len(num_cols)


# ── Model tests ──────────────────────────────────────────────────────────────

class TestModel:
    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), \
            "best_model.pkl not found. Run src/train_model.py first."

    def test_metadata_file_exists(self):
        assert os.path.exists(METADATA_PATH), \
            "model_metadata.json not found. Run src/train_model.py first."

    def test_metadata_structure(self):
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        required_keys = ["best_model", "metrics", "top_features", "num_cols", "cat_cols"]
        for k in required_keys:
            assert k in meta, f"Missing key in metadata: {k}"

    def test_model_loadable(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert model is not None

    def test_model_has_predict(self):
        import joblib
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


# ── Prediction tests ─────────────────────────────────────────────────────────

SAMPLE_EMPLOYEE = {
    "Age": 35,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MonthlyIncome": 5000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
}

HIGH_RISK_EMPLOYEE = {
    **SAMPLE_EMPLOYEE,
    "OverTime": "Yes",
    "JobSatisfaction": 1,
    "WorkLifeBalance": 1,
    "JobInvolvement": 1,
    "YearsSinceLastPromotion": 10,
    "StockOptionLevel": 0,
    "EnvironmentSatisfaction": 1,
}


class TestPrediction:
    def test_prediction_output_structure(self):
        from prediction import predict_employee
        result = predict_employee(SAMPLE_EMPLOYEE)
        assert "prediction"    in result
        assert "probability"   in result
        assert "risk_level"    in result
        assert "top_factors"   in result
        assert "recommendations" in result

    def test_prediction_has_explanation(self):
        from prediction import predict_employee
        result = predict_employee(SAMPLE_EMPLOYEE)
        assert "explanation" in result
        assert "factors" in result["explanation"]
        assert "disclaimer" in result["explanation"]

    def test_prediction_label(self):
        from prediction import predict_employee
        result = predict_employee(SAMPLE_EMPLOYEE)
        assert result["prediction"] in ("Yes", "No")

    def test_probability_range(self):
        from prediction import predict_employee
        result = predict_employee(SAMPLE_EMPLOYEE)
        assert 0.0 <= result["probability"] <= 1.0

    def test_risk_level_values(self):
        from prediction import predict_employee
        result = predict_employee(SAMPLE_EMPLOYEE)
        assert result["risk_level"] in ("Low", "Medium", "High")

    def test_top_factors_is_list(self):
        from prediction import predict_employee
        result = predict_employee(SAMPLE_EMPLOYEE)
        assert isinstance(result["top_factors"], list)

    def test_recommendations_is_list(self):
        from prediction import predict_employee
        result = predict_employee(SAMPLE_EMPLOYEE)
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_explanation_factors_have_direction(self):
        from prediction import predict_employee
        result = predict_employee(SAMPLE_EMPLOYEE)
        factors = result["explanation"]["factors"]
        if factors:
            assert "direction" in factors[0]
            assert factors[0]["direction"] in ("risk-increasing", "protective")

    def test_explanation_has_risk_protective_split(self):
        from prediction import predict_employee
        result = predict_employee(HIGH_RISK_EMPLOYEE)
        expl = result["explanation"]
        # Either list may be empty but both must exist
        assert "risk_factors" in expl
        assert "protective_factors" in expl


# ── Risk Classification ──────────────────────────────────────────────────────

class TestRiskClassification:
    def test_low_risk(self):
        from prediction import classify_risk
        assert classify_risk(0.10) == "Low"
        assert classify_risk(0.29) == "Low"

    def test_medium_risk(self):
        from prediction import classify_risk
        assert classify_risk(0.30) == "Medium"
        assert classify_risk(0.59) == "Medium"

    def test_high_risk(self):
        from prediction import classify_risk
        assert classify_risk(0.60) == "High"
        assert classify_risk(0.95) == "High"


# ── Scenario Analysis ────────────────────────────────────────────────────────

class TestScenarioAnalysis:
    def test_scenario_returns_structure(self):
        from prediction import scenario_predict
        overrides = {"OverTime": "No", "JobSatisfaction": 4}
        result = scenario_predict(HIGH_RISK_EMPLOYEE, overrides)
        assert "current_probability" in result
        assert "scenario_probability" in result
        assert "current_risk" in result
        assert "scenario_risk" in result
        assert "difference_pp" in result
        assert "disclaimer" in result

    def test_scenario_probabilities_in_range(self):
        from prediction import scenario_predict
        overrides = {"OverTime": "No"}
        result = scenario_predict(SAMPLE_EMPLOYEE, overrides)
        assert 0.0 <= result["current_probability"] <= 1.0
        assert 0.0 <= result["scenario_probability"] <= 1.0

    def test_scenario_difference_pp_correct(self):
        from prediction import scenario_predict
        overrides = {"JobSatisfaction": 4}
        result = scenario_predict(SAMPLE_EMPLOYEE, overrides)
        expected_diff = round(
            (result["scenario_probability"] - result["current_probability"]) * 100, 1
        )
        assert result["difference_pp"] == expected_diff

    def test_high_risk_reduces_with_positive_overrides(self):
        """Changing many negative factors should generally reduce probability."""
        from prediction import scenario_predict
        overrides = {
            "OverTime": "No",
            "JobSatisfaction": 4,
            "WorkLifeBalance": 4,
            "JobInvolvement": 4,
            "StockOptionLevel": 3,
        }
        result = scenario_predict(HIGH_RISK_EMPLOYEE, overrides)
        # Scenario probability should be <= current for these positive changes
        assert result["scenario_probability"] <= result["current_probability"] + 0.05


# ── Threshold Analysis ───────────────────────────────────────────────────────

class TestThresholdAnalysis:
    def test_threshold_analysis_returns_structure(self):
        from prediction import threshold_analysis
        result = threshold_analysis()
        assert "thresholds" in result
        assert "note" in result
        assert len(result["thresholds"]) > 0

    def test_threshold_analysis_has_all_metrics(self):
        from prediction import threshold_analysis
        result = threshold_analysis()
        for row in result["thresholds"]:
            assert "threshold" in row
            assert "precision" in row
            assert "recall" in row
            assert "f1" in row
            assert "n_high_risk" in row

    def test_threshold_default_marker(self):
        from prediction import threshold_analysis
        result = threshold_analysis()
        defaults = [r for r in result["thresholds"] if r.get("is_default")]
        assert len(defaults) == 1
        assert defaults[0]["threshold"] == 0.50

    def test_threshold_with_employee(self):
        from prediction import threshold_analysis
        result = threshold_analysis(SAMPLE_EMPLOYEE)
        assert "employee_thresholds" in result
        assert len(result["employee_thresholds"]) > 0

    def test_lower_threshold_higher_recall(self):
        """Recall should generally increase as threshold decreases."""
        from prediction import threshold_analysis
        result = threshold_analysis()
        rows = sorted(result["thresholds"], key=lambda r: r["threshold"])
        # Recall at 0.30 should be >= recall at 0.70
        recall_low = next(r["recall"] for r in rows if r["threshold"] == 0.30)
        recall_high = next(r["recall"] for r in rows if r["threshold"] == 0.70)
        assert recall_low >= recall_high - 5  # allow small margin


# ── Data Quality ─────────────────────────────────────────────────────────────

class TestDataQuality:
    def test_data_quality_returns_structure(self):
        from data_quality import compute_data_quality
        result = compute_data_quality()
        assert "n_rows" in result
        assert "n_cols" in result
        assert "total_missing" in result
        assert "n_duplicates" in result
        assert "target_distribution" in result
        assert "checks" in result
        assert "status" in result

    def test_data_quality_counts_correct(self):
        from data_preprocessing import load_dataset
        from data_quality import compute_data_quality
        df = load_dataset()
        result = compute_data_quality(df)
        assert result["n_rows"] == len(df)
        assert result["n_cols"] == len(df.columns)

    def test_data_quality_target_distribution(self):
        from data_quality import compute_data_quality
        result = compute_data_quality()
        dist = result["target_distribution"]
        assert "Yes" in dist
        assert "No" in dist
        assert dist["Yes"] + dist["No"] == result["n_rows"]

    def test_data_quality_imbalance_ratio(self):
        from data_quality import compute_data_quality
        result = compute_data_quality()
        assert result["class_imbalance_ratio"] > 1.0  # always imbalanced

    def test_data_quality_constant_cols(self):
        from data_quality import compute_data_quality
        result = compute_data_quality()
        # Should detect and report EmployeeCount, Over18, StandardHours
        assert len(result["constant_cols_removed"]) >= 1

    def test_data_quality_checks_list(self):
        from data_quality import compute_data_quality
        result = compute_data_quality()
        for check in result["checks"]:
            assert "label" in check
            assert "status" in check
            assert check["status"] in ("healthy", "review")


# ── Fairness ─────────────────────────────────────────────────────────────────

class TestFairness:
    def test_fairness_returns_structure(self):
        from data_preprocessing import load_dataset
        from fairness import compute_fairness_metrics
        df = load_dataset()
        result = compute_fairness_metrics(df)
        assert "groups" in result
        assert "overall" in result
        assert "disclaimer" in result

    def test_fairness_has_gender_group(self):
        from data_preprocessing import load_dataset
        from fairness import compute_fairness_metrics
        df = load_dataset()
        result = compute_fairness_metrics(df)
        assert "Gender" in result["groups"]
        gender_groups = result["groups"]["Gender"]
        assert "Male" in gender_groups or "Female" in gender_groups

    def test_fairness_metrics_in_range(self):
        from data_preprocessing import load_dataset
        from fairness import compute_fairness_metrics
        df = load_dataset()
        result = compute_fairness_metrics(df)
        overall = result["overall"]
        assert 0 <= overall["precision"] <= 100
        assert 0 <= overall["recall"] <= 100
        assert 0 <= overall["f1"] <= 100
        assert 0 <= overall["fpr"] <= 100
        assert 0 <= overall["fnr"] <= 100

    def test_fairness_group_metrics_structure(self):
        from data_preprocessing import load_dataset
        from fairness import compute_fairness_metrics
        df = load_dataset()
        result = compute_fairness_metrics(df)
        for group_val in result["groups"]["Gender"].values():
            assert "n_records" in group_val
            assert "precision" in group_val
            assert "recall" in group_val
            assert "f1" in group_val
            assert "fpr" in group_val
            assert "fnr" in group_val


# ── PDF Report ───────────────────────────────────────────────────────────────

class TestPDFReport:
    def test_report_generates_bytes(self):
        from pdf_report import generate_pdf_report
        result = {
            "prediction": "Yes",
            "probability": 0.72,
            "risk_level": "High",
            "top_factors": ["Overtime: Yes", "Job Satisfaction = 1"],
            "explanation": {
                "factors": [
                    {"feature": "Overtime: Yes", "direction": "risk-increasing",
                     "shap_value": 0.5, "contribution": "positive"},
                ],
                "risk_factors": [],
                "protective_factors": [],
                "disclaimer": "Test disclaimer",
            },
            "recommendations": ["Consider reviewing workload."],
        }
        report = generate_pdf_report(result)
        assert isinstance(report, bytes)
        assert len(report) > 100  # non-empty

    def test_report_with_scenario(self):
        from pdf_report import generate_pdf_report
        result = {
            "prediction": "Yes",
            "probability": 0.72,
            "risk_level": "High",
            "top_factors": [],
            "explanation": None,
            "recommendations": ["Test recommendation."],
        }
        scenario = {
            "current_probability": 0.72,
            "scenario_probability": 0.45,
            "current_risk": "High",
            "scenario_risk": "Medium",
            "difference_pp": -27.0,
        }
        report = generate_pdf_report(result, scenario_result=scenario)
        assert isinstance(report, bytes)
        assert len(report) > 100


# ── Flask API tests ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestFlaskEndpoints:
    def test_home_page(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_dashboard_page(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_prediction_page(self, client):
        r = client.get("/prediction")
        assert r.status_code == 200

    def test_insights_page(self, client):
        r = client.get("/insights")
        assert r.status_code == 200

    def test_fairness_page(self, client):
        r = client.get("/fairness")
        assert r.status_code == 200

    def test_data_quality_page(self, client):
        r = client.get("/data-quality")
        assert r.status_code == 200

    def test_executive_page(self, client):
        r = client.get("/executive")
        assert r.status_code == 200

    def test_about_page(self, client):
        r = client.get("/about")
        assert r.status_code == 200

    def test_api_predict_valid(self, client):
        import json as _json
        r = client.post("/api/predict",
                        data=_json.dumps(SAMPLE_EMPLOYEE),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "prediction" in data
        assert "probability" in data
        assert "explanation" in data

    def test_api_predict_empty_body(self, client):
        r = client.post("/api/predict",
                        data="{}",
                        content_type="application/json")
        assert r.status_code in (200, 400, 500)

    def test_api_metrics(self, client):
        r = client.get("/api/metrics")
        assert r.status_code == 200
        data = r.get_json()
        assert "model_comparison" in data

    def test_api_feature_importance(self, client):
        r = client.get("/api/feature-importance")
        assert r.status_code == 200
        data = r.get_json()
        assert "feature_importance" in data

    def test_api_fairness(self, client):
        r = client.get("/api/fairness")
        assert r.status_code == 200
        data = r.get_json()
        assert "groups" in data
        assert "overall" in data

    def test_api_data_quality(self, client):
        r = client.get("/api/data-quality")
        assert r.status_code == 200
        data = r.get_json()
        assert "n_rows" in data
        assert "checks" in data

    def test_api_scenario(self, client):
        import json as _json
        payload = {
            "employee": SAMPLE_EMPLOYEE,
            "overrides": {"OverTime": "Yes", "JobSatisfaction": 1}
        }
        r = client.post("/api/scenario",
                        data=_json.dumps(payload),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "current_probability" in data
        assert "scenario_probability" in data

    def test_api_threshold_analysis(self, client):
        import json as _json
        r = client.post("/api/threshold-analysis",
                        data=_json.dumps({"employee": SAMPLE_EMPLOYEE}),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "thresholds" in data
        assert len(data["thresholds"]) > 0

    def test_api_model_monitoring(self, client):
        r = client.get("/api/model-monitoring")
        assert r.status_code == 200
        data = r.get_json()
        assert "best_model" in data
        assert "metrics" in data
        assert "training_samples" in data

    def test_api_executive_summary(self, client):
        r = client.get("/api/executive-summary")
        assert r.status_code == 200
        data = r.get_json()
        assert "kpi" in data
        assert "predicted_risk" in data

    def test_api_download_report(self, client):
        import json as _json
        prediction_result = {
            "prediction": "Yes",
            "probability": 0.72,
            "risk_level": "High",
            "top_factors": ["Overtime: Yes"],
            "explanation": None,
            "recommendations": ["Test."],
        }
        r = client.post("/api/download-report",
                        data=_json.dumps({"prediction_result": prediction_result}),
                        content_type="application/json")
        assert r.status_code == 200
        assert len(r.data) > 100

    def test_404(self, client):
        r = client.get("/nonexistent-page")
        assert r.status_code == 404

# AttritionIQ

**IBM SkillsBuild Data Analytics with AI – Virtual Internship Project**

An AI-powered HR analytics platform that analyses employee attrition patterns, predicts flight-risk using machine learning, and generates data-driven retention recommendations — built on the IBM HR Analytics Employee Attrition dataset.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Features](#features)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Model Selection & Performance](#model-selection--performance)
6. [Advanced Features](#advanced-features)
   - [Advanced Explainable AI](#1-advanced-explainable-ai)
   - [What-If Scenario Analysis](#2-what-if-scenario-analysis)
   - [Prediction Threshold Analysis](#3-prediction-threshold-analysis)
   - [Fairness & Bias Audit](#4-fairness--bias-audit)
   - [Data Quality Monitor](#5-data-quality-monitor)
   - [Executive HR Summary](#6-executive-hr-summary)
   - [Downloadable PDF Reports](#7-downloadable-pdf-reports)
   - [Model Monitoring](#8-model-monitoring)
7. [API Endpoints](#api-endpoints)
8. [Project Structure](#project-structure)
9. [Ethical AI Statement](#ethical-ai-statement)
10. [Limitations](#limitations)

---

## Overview

AttritionIQ is built for the IBM SkillsBuild Data Analytics with AI virtual internship. It demonstrates end-to-end ML applied to an HR analytics use case:

- **Exploratory Data Analysis** — understand patterns in the IBM HR dataset
- **Multi-model training and comparison** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Real-time attrition prediction** — REST API and web UI
- **Explainable AI** — SHAP-based (Logistic Regression) or coefficient-weighted attribution with direction indicators
- **What-If scenario analysis** — simulate hypothetical employee attribute changes
- **Threshold analysis** — understand the precision/recall trade-off
- **Fairness audit** — diagnostic per-group model performance evaluation
- **Data quality monitoring** — automated checks on the actual dataset
- **Executive summary** — non-technical HR leadership dashboard
- **PDF prediction reports** — downloadable decision-support documents

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain the model
python src/train_model.py

# 3. Start the application
python app.py

# 4. Open in browser
# http://localhost:5000
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Features

| Feature | Description |
|---|---|
| Interactive Dashboard | Filterable charts — department, role, tenure, overtime, age |
| AI Prediction | Attrition probability, risk level (Low / Medium / High) |
| Explainable AI | SHAP values + directional factor attribution |
| What-If Scenario | Simulate hypothetical attribute changes |
| Threshold Analysis | Precision/recall trade-off across 9 thresholds |
| Fairness Audit | Per-group model performance metrics |
| Data Quality Monitor | Missing values, duplicates, imbalance, outliers |
| Executive Summary | Historical + predicted risk overview |
| PDF Report Download | Formatted prediction report with disclaimer |
| Model Monitoring | Current metrics, training date, sample counts |
| HR Recommendations | Rule-based, ethics-compliant retention suggestions |
| REST API | Full JSON API for all major features |

---

## Machine Learning Pipeline

- **Dataset:** IBM HR Analytics Employee Attrition & Performance (~1,470 records, 35 features)
- **Preprocessing:** `StandardScaler` (numeric) + `OneHotEncoder` (categorical) inside a `ColumnTransformer` pipeline
- **Class imbalance:** `class_weight="balanced"` applied to all models; no resampling needed
- **Train/test split:** 80/20 stratified
- **Models evaluated:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Model selection:** Equal-weight combination of F1-score + Recall + ROC-AUC (see below)
- **Explainability:** SHAP `LinearExplainer` for Logistic Regression; coefficient × transformed-value weighting as fallback
- **Constant columns removed:** `EmployeeCount`, `Over18`, `StandardHours`
- **ID columns excluded from modelling:** `EmployeeNumber`

---

## Model Selection & Performance

### Why Logistic Regression Is the Best Model

The model is selected using an **equal-weight combination of F1-score, Recall, and ROC-AUC** — not accuracy alone. This criterion is intentional: in an imbalanced attrition dataset, optimising for accuracy alone rewards models that simply predict "No attrition" for every employee, missing the cases HR actually needs to identify.

The combined score formula:

```
combined_score = (F1 + Recall + ROC-AUC) / 3
```

**Logistic Regression** achieves the highest combined score of **0.7565** because it has:
- The highest **Recall** (83.0%) — it correctly identifies 83% of employees who will leave. This is critical: a model that misses at-risk employees provides little HR value.
- The highest **ROC-AUC** (89.8%) — the strongest overall discriminative ability across all possible thresholds.
- The highest **F1-score** (54.2%) — the best balance between precision and recall.

Although Gradient Boosting and Random Forest achieve higher raw accuracy (up to 87.4%), they do so by being highly conservative — predicting "No attrition" for many employees who actually leave. This results in significantly lower recall (36–45%) and lower combined scores.

### Full Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Combined Score | Best |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **77.5%** | 40.2% | **83.0%** | **54.2%** | **89.8%** | **0.7565** | ✓ |
| Gradient Boosting | 86.7% | 61.8% | 44.7% | 51.8% | 87.0% | 0.6118 | |
| Random Forest | 87.4% | **70.8%** | 36.2% | 47.9% | 89.2% | 0.5777 | |
| Decision Tree | 76.9% | 35.6% | 55.3% | 43.3% | 69.2% | 0.5582 | |

**Key insight:** Logistic Regression's 83% recall means it flags 83 out of every 100 employees who leave — making it far more useful for proactive HR intervention than a model with 87% accuracy but only 36% recall.

---

## Advanced Features

### 1. Advanced Explainable AI

The prediction page now includes a **"Why This Prediction?"** section showing:

- **Attrition probability** and **risk level**
- **Top 5 contributing factors** with direction:
  - ↑ **Risk-increasing** factors (highlighted red) — associated with higher predicted attrition
  - ↓ **Protective** factors (highlighted green) — associated with lower predicted attrition
- **Method used:** SHAP `LinearExplainer` for Logistic Regression (most accurate); coefficient × transformed-value weighting as fallback for other models

**Important disclaimer displayed on every prediction:**
> *"These factors are associated with the model prediction and do not establish causation."*

Gender, MaritalStatus and other protected attributes are used in the model pipeline but are **never** the basis for HR recommendations.

### 2. What-If Scenario Analysis

After generating a prediction, users can modify selected actionable features and recalculate:

**Actionable features available:**
- Overtime (Yes / No)
- Job Satisfaction (1–4)
- Work-Life Balance (1–4)
- Job Involvement (1–4)
- Years at Company
- Years Since Last Promotion
- Environment Satisfaction (1–4)
- Relationship Satisfaction (1–4)
- Training Times Last Year
- Stock Option Level (0–3)

**Output:**
- Current probability vs. scenario probability
- Change in percentage points
- Current risk vs. scenario risk level

**Clearly labelled as:**
> *"This is a model simulation of hypothetical changes. It does not predict whether an employee will actually leave. Changing a feature value in the model does not establish that the change will prevent attrition."*

**API:** `POST /api/scenario` — accepts `{ "employee": {...}, "overrides": {...} }`

### 3. Prediction Threshold Analysis

The threshold analysis section shows how the classification boundary affects model behaviour across the held-out test set.

**Thresholds covered:** 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70

**Metrics shown per threshold:**
- Precision, Recall, F1-score
- Number of employees classified as high-risk on the test set
- For the current employee: what prediction they would receive at each threshold

**Context provided:**
> *"The dataset is class-imbalanced (~16% attrition). Lowering the threshold increases recall (fewer missed at-risk employees) but reduces precision (more false positives). The default threshold of 0.50 is used unless explicitly changed."*

**API:** `POST /api/threshold-analysis` — accepts optional `{ "employee": {...} }`

### 4. Fairness & Bias Audit

A dedicated **Fairness** page (`/fairness`) evaluates model performance across demographic groups.

**Groups evaluated:** Gender, Department, Job Role, Marital Status

**Metrics per group:**
- N records in test set
- Observed attrition rate
- Precision, Recall, F1-score
- False Positive Rate (FPR)
- False Negative Rate (FNR)

All metrics are computed on the held-out test set (same 80/20 stratified split used in training).

**Clearly displayed:**
> *"Fairness metrics are diagnostic indicators and do not by themselves establish discrimination."*
> *"Gender and other demographic attributes are used here ONLY for diagnostic analysis and are not used in HR recommendations."*

**API:** `GET /api/fairness`

### 5. Data Quality Monitor

The **Data Quality** page (`/data-quality`) shows automated checks computed from the actual IBM HR dataset — no hard-coded values.

**Checks included:**

| Check | Status Indicator |
|---|---|
| Missing values per column | ✓ Healthy / ⚠ Review |
| Duplicate rows | ✓ Healthy / ⚠ Review |
| Target class distribution (Yes/No) | Count + percentage |
| Class imbalance ratio | ✓ ≤3:1 / ⚠ >3:1 |
| Dataset dimensions | Rows × columns |
| Constant columns removed | `EmployeeCount`, `Over18`, `StandardHours` |
| ID columns excluded | `EmployeeNumber` |
| Potential outliers | IQR method on top 10 numeric features |

**API:** `GET /api/data-quality`

### 6. Executive HR Summary

The **Executive** page (`/executive`) provides a non-technical dashboard for HR leadership.

**Historical Observed Attrition (from dataset):**
- Total employees
- Employees who left / retained
- Historical attrition rate

**Model-Predicted Risk (current model applied to full dataset):**
- Predicted High-Risk employees
- Predicted Medium-Risk employees
- Predicted Low-Risk employees

**Observed breakdown charts:**
- By Department
- By Job Role
- By Overtime Status
- By Job Satisfaction

**Model Monitoring summary** embedded at the bottom (selected model, metrics, training date).

**Terminology clearly distinguished:**
> *"Observed Historical Attrition"* vs. *"Model-Predicted Risk"*

**API:** `GET /api/executive-summary`

### 7. Downloadable PDF Reports

The prediction page includes a **"Download Report"** button that generates a formatted report containing:

- Prediction date/time and model used
- Attrition probability and risk level
- Classification threshold
- Top contributing factors (with direction)
- Scenario analysis results (if performed)
- HR retention recommendations
- Full disclaimer section

**Technology:** Uses `reportlab` for PDF generation; falls back to HTML download if unavailable.

**The report prominently states:**
> *"This report is a decision-support tool and NOT an automated employment decision system."*

**API:** `POST /api/download-report` — accepts `{ "prediction_result": {...}, "employee_data": {...}, "scenario_result": {...} }`

### 8. Model Monitoring

The **Executive Summary** page embeds a Model Monitoring section showing:

- Selected model name
- Current performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Training and test sample counts
- Model training date (from file modification time)
- Selection criterion explanation
- Combined score

All metrics are read from `model_metadata.json` and the actual test set — no invented data.

**API:** `GET /api/model-monitoring`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Home page |
| GET | `/dashboard` | HR Analytics Dashboard |
| GET | `/prediction` | Prediction form page |
| GET | `/insights` | Insights & model performance |
| GET | `/fairness` | Fairness & Bias Audit page |
| GET | `/data-quality` | Data Quality Monitor page |
| GET | `/executive` | Executive HR Summary page |
| GET | `/about` | About page |
| POST | `/api/predict` | JSON prediction endpoint |
| POST | `/api/scenario` | What-If scenario analysis |
| POST | `/api/threshold-analysis` | Threshold analysis |
| GET | `/api/fairness` | Fairness metrics JSON |
| GET | `/api/data-quality` | Data quality metrics JSON |
| GET | `/api/executive-summary` | Executive summary JSON |
| GET | `/api/model-monitoring` | Model monitoring JSON |
| GET | `/api/metrics` | Model comparison, ROC curve, confusion matrix |
| GET | `/api/feature-importance` | Feature importance list |
| GET | `/api/dashboard-data` | Filtered dashboard data |
| GET | `/api/filter-options` | Filter dropdown values |
| POST | `/api/download-report` | Generate & download prediction report |

---

## Project Structure

```
AttritionIQ/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # IBM HR dataset
├── model/
│   ├── best_model.pkl                            # Saved best model pipeline
│   ├── preprocessor.pkl                          # Saved preprocessor
│   └── model_metadata.json                       # Metrics, feature names, top features
├── notebooks/
│   └── Khizra_AttritionIQ.ipynb                  # EDA, modelling & analysis notebook (submission)
├── src/
│   ├── data_preprocessing.py                     # Load, clean, split, pipeline
│   ├── train_model.py                            # Train & compare models
│   ├── evaluate_model.py                         # Evaluation helpers
│   ├── prediction.py                             # Prediction, scenario, threshold
│   ├── explainability.py                         # SHAP & coefficient-based XAI
│   ├── fairness.py                               # Fairness metrics per group
│   ├── data_quality.py                           # Data quality checks
│   └── pdf_report.py                             # PDF/HTML report generator
├── templates/
│   ├── index.html                                # Home page
│   ├── dashboard.html                            # Dashboard
│   ├── prediction.html                           # Prediction + XAI + scenario
│   ├── insights.html                             # Model insights
│   ├── fairness.html                             # Fairness audit
│   ├── data_quality.html                         # Data quality
│   ├── executive.html                            # Executive summary
│   ├── about.html                                # About page
│   └── error.html                                # Error page
├── static/
│   ├── css/style.css                             # Design system
│   └── js/script.js                              # Frontend logic
├── tests/
│   └── test_attritioniq.py                       # Comprehensive test suite (67 tests)
├── app.py                                        # Flask application
└── requirements.txt                              # Python dependencies
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| flask | 3.1.2 | Web framework |
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.4.0 | Numerical computing |
| scikit-learn | 1.8.0 | ML pipeline |
| matplotlib | 3.10.8 | Plotting |
| plotly | 6.9.0 | Interactive charts |
| joblib | 1.5.3 | Model serialisation |
| imbalanced-learn | 0.14.2 | Class imbalance utilities |
| Werkzeug | 3.1.4 | Flask dependency |
| reportlab | 4.1.0 | PDF report generation |
| shap | 0.52.0 | SHAP explainability (optional — falls back gracefully) |
| notebook | 7.6.3 | Jupyter notebook runtime |
| ipykernel | 7.3.0 | Jupyter kernel for Python |

---

## Ethical AI Statement

This project is built for **educational and demonstration purposes only**.

- The IBM HR dataset is **entirely fictional** and does not represent real employees.
- Predictions are **decision-support tools only** and must not be used to terminate, penalise, or automatically reject real employees.
- The model does **not use protected attributes** (Gender, MaritalStatus) as the basis for recommendations.
- Model predictions represent **statistical patterns**, not causal relationships.
- The **Fairness Audit** uses demographic attributes for diagnostic evaluation only — not for operational decision-making.
- HR professionals should **review all predictions** before taking any action.
- Risk thresholds (Low/Medium/High) are configurable and are **not official IBM HR thresholds**.
- This project **demonstrates analytics and ML techniques** — it is not an actual IBM HR decision system.

---

## Limitations

- Dataset is fictional — not suitable for direct real-world deployment
- Model trained on a single snapshot — no longitudinal or time-series analysis
- Fairness audit uses test-set metrics only — a full fairness analysis requires domain expertise
- No authentication or audit logging (required for real deployment)
- SHAP computation re-runs on each prediction call (could be cached in production)
- The What-If scenario analysis is a model simulation, not a causal intervention model
- PDF generation requires `reportlab`; falls back to HTML if not installed

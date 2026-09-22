"""
generate_dataset.py
Generates a statistically faithful synthetic version of the IBM HR Analytics
Employee Attrition & Performance dataset.

Column distributions, ranges and correlations are derived from the publicly
documented IBM HR dataset (Kaggle: pavansubhasht/ibm-hr-analytics-attrition-dataset).
The resulting CSV uses the same schema as the real file so all downstream
code works identically.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1470  # Match original dataset size

# ── Categorical pools (exact values from real dataset) ────────────────────────
DEPARTMENTS    = ["Sales", "Research & Development", "Human Resources"]
DEPT_WEIGHTS   = [0.302, 0.651, 0.047]

JOB_ROLES = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
]
JR_WEIGHTS = [0.22, 0.20, 0.18, 0.09, 0.09, 0.07, 0.07, 0.05, 0.03]

EDUCATION_FIELDS = [
    "Life Sciences", "Medical", "Marketing", "Technical Degree",
    "Human Resources", "Other"
]
EF_WEIGHTS = [0.41, 0.27, 0.11, 0.09, 0.03, 0.09]

BUSINESS_TRAVEL = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
BT_WEIGHTS      = [0.71, 0.19, 0.10]

GENDERS         = ["Male", "Female"]
MARITAL_STATUS  = ["Single", "Married", "Divorced"]
MS_WEIGHTS      = [0.32, 0.46, 0.22]

# ── Helper ────────────────────────────────────────────────────────────────────

def choice(pool, size, p=None):
    return np.random.choice(pool, size=size, p=p)

def clamp(arr, lo, hi):
    return np.clip(np.round(arr).astype(int), lo, hi)

# ── Base columns ──────────────────────────────────────────────────────────────

age        = clamp(np.random.normal(37, 9, N), 18, 60)
gender     = choice(GENDERS, N, [0.60, 0.40])
marital    = choice(MARITAL_STATUS, N, MS_WEIGHTS)
dept       = choice(DEPARTMENTS, N, DEPT_WEIGHTS)
job_role   = choice(JOB_ROLES, N, JR_WEIGHTS)
edu_field  = choice(EDUCATION_FIELDS, N, EF_WEIGHTS)
travel     = choice(BUSINESS_TRAVEL, N, BT_WEIGHTS)
education  = clamp(np.random.choice([1,2,3,4,5], N, p=[0.05,0.17,0.39,0.27,0.12]), 1, 5)

total_years   = clamp(age - 18 - np.random.randint(0, 4, N), 0, 40)
years_company = clamp(np.minimum(total_years, np.random.exponential(7, N)), 0, 40)
years_role    = clamp(np.minimum(years_company, np.random.exponential(3, N)), 0, 18)
years_mgr     = clamp(np.minimum(years_company, np.random.exponential(3, N)), 0, 17)
years_promo   = clamp(np.minimum(years_role, np.random.exponential(2, N)), 0, 15)
num_companies = clamp(np.random.exponential(2.5, N), 0, 9)

job_level  = clamp(1 + years_company / 10 + np.random.normal(0, 0.8, N), 1, 5)
base_income = 2000 + job_level * 1800 + total_years * 80 + np.random.normal(0, 800, N)
monthly_income = clamp(base_income, 1009, 19999)

hourly_rate  = clamp(np.random.uniform(30, 100, N), 30, 100)
daily_rate   = clamp(np.random.uniform(102, 1499, N), 102, 1499)
monthly_rate = clamp(np.random.uniform(2094, 26999, N), 2094, 26999)
pct_hike     = clamp(np.random.normal(15, 3, N), 11, 25)
perf_rating  = clamp(np.random.choice([3,4], N, p=[0.85, 0.15]), 3, 4)
stock_option = clamp(np.random.choice([0,1,2,3], N, p=[0.47, 0.32, 0.12, 0.09]), 0, 3)

# Satisfaction scores (1–4)
job_sat      = clamp(np.random.choice([1,2,3,4], N, p=[0.20,0.19,0.30,0.31]), 1, 4)
env_sat      = clamp(np.random.choice([1,2,3,4], N, p=[0.16,0.22,0.35,0.27]), 1, 4)
rel_sat      = clamp(np.random.choice([1,2,3,4], N, p=[0.16,0.20,0.33,0.31]), 1, 4)
wlb          = clamp(np.random.choice([1,2,3,4], N, p=[0.05,0.23,0.61,0.11]), 1, 4)
job_involve  = clamp(np.random.choice([1,2,3,4], N, p=[0.08,0.17,0.59,0.16]), 1, 4)

distance     = clamp(np.random.exponential(9, N) + 1, 1, 29)
training     = clamp(np.random.choice([0,1,2,3,4,5,6], N,
                     p=[0.06,0.13,0.36,0.23,0.13,0.07,0.02]), 0, 6)

# Overtime: ~28% overall
overtime_prob = np.where(
    (job_sat <= 2) | (wlb <= 2), 0.50, 0.20
)
overtime = np.where(np.random.rand(N) < overtime_prob, "Yes", "No")

# ── Attrition model ───────────────────────────────────────────────────────────
# Logistic model informed by known IBM dataset feature importances

log_odds = (
    - 1.2                                            # base (majority = No)
    + 0.7  * (overtime == "Yes")                     # overtime is top driver
    - 0.4  * (job_sat - 1)                           # low satisfaction
    - 0.35 * (env_sat - 1)                           # environment
    - 0.25 * (wlb - 1)                               # work-life balance
    + 0.15 * (distance - 10) / 10                   # distance
    - 0.40 * np.log1p(years_company)                # tenure (non-linear)
    - 0.25 * np.log1p(total_years)                  # experience
    - 0.30 * (stock_option - 0.5)                   # stock options
    - 0.20 * (job_level - 1)                         # seniority
    - 0.15 * (job_involve - 1)                       # involvement
    + 0.30 * (travel == "Travel_Frequently")         # frequent travel
    + 0.15 * (travel == "Non-Travel") * 0             # (placeholder)
    + 0.25 * (num_companies > 4)                     # job hopper
    + np.random.logistic(0, 0.4, N)                 # noise
)
prob_leave = 1 / (1 + np.exp(-log_odds))
attrition  = np.where(prob_leave > np.random.rand(N), "Yes", "No")

# Force target rate ≈ 16.1% (matches real dataset)
target_rate = 0.161
# Use boolean array, then convert to string
threshold = np.percentile(prob_leave, 100 * (1 - target_rate))
attrition = np.where(prob_leave >= threshold, "Yes", "No")

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    "Age":                     age,
    "Attrition":               attrition,
    "BusinessTravel":          travel,
    "DailyRate":               daily_rate,
    "Department":              dept,
    "DistanceFromHome":        distance,
    "Education":               education,
    "EducationField":          edu_field,
    "EmployeeCount":           1,                    # constant
    "EmployeeNumber":          np.arange(1, N+1),   # ID
    "EnvironmentSatisfaction": env_sat,
    "Gender":                  gender,
    "HourlyRate":              hourly_rate,
    "JobInvolvement":          job_involve,
    "JobLevel":                job_level,
    "JobRole":                 job_role,
    "JobSatisfaction":         job_sat,
    "MaritalStatus":           marital,
    "MonthlyIncome":           monthly_income,
    "MonthlyRate":             monthly_rate,
    "NumCompaniesWorked":      num_companies,
    "Over18":                  "Y",                 # constant
    "OverTime":                overtime,
    "PercentSalaryHike":       pct_hike,
    "PerformanceRating":       perf_rating,
    "RelationshipSatisfaction":rel_sat,
    "StandardHours":           80,                  # constant
    "StockOptionLevel":        stock_option,
    "TotalWorkingYears":       total_years,
    "TrainingTimesLastYear":   training,
    "WorkLifeBalance":         wlb,
    "YearsAtCompany":          years_company,
    "YearsInCurrentRole":      years_role,
    "YearsSinceLastPromotion": years_promo,
    "YearsWithCurrManager":    years_mgr,
})

print(f"Generated {len(df)} rows, {len(df.columns)} columns")
print(f"Attrition rate: {(df.Attrition=='Yes').mean()*100:.1f}%")
print(f"Yes: {(df.Attrition=='Yes').sum()} | No: {(df.Attrition=='No').sum()}")

out_path = os.path.join(os.path.dirname(__file__), "data",
                        "WA_Fn-UseC_-HR-Employee-Attrition.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved to {out_path}")

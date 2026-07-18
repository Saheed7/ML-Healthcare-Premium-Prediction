"""
prediction_helper.py
Preprocessing + prediction for the Healthcare Premium Prediction app.
Matches the FIXED training notebook exactly (snake_case columns).
Expects: artifacts/model.joblib  and  artifacts/scaler.joblib
"""

from pathlib import Path
import pandas as pd
from joblib import load

BASE_DIR = Path(__file__).parent

model = load(BASE_DIR / "artifacts" / "model.joblib")

scaler_bundle = load(BASE_DIR / "artifacts" / "scaler.joblib")
scaler = scaler_bundle["scaler"]
cols_to_scale = scaler_bundle["cols_to_scale"]
# cols_to_scale = ['age', 'number_of_dependants', 'income_level',
#                  'income_lakhs', 'insurance_plan', 'lifestyle_risk_score']

# Final feature order the model was trained on (X_reduced.columns)
MODEL_FEATURES = [
    "age", "number_of_dependants", "income_lakhs", "insurance_plan",
    "normalized_risk_score", "lifestyle_risk_score",
    "gender_Male",
    "region_Northwest", "region_Southeast", "region_Southwest",
    "marital_status_Unmarried",
    "bmi_category_Obesity", "bmi_category_Overweight", "bmi_category_Underweight",
    "smoking_status_Occasional", "smoking_status_Regular",
    "employment_status_Salaried", "employment_status_Self-Employed",
]

# Keys main.py must send in input_dict (UI-facing names)
EXPECTED_KEYS = [
    "age", "number_of_dependants", "income_lakhs", "insurance_plan",
    "medical_history", "physical_activity", "stress_level",
    "gender", "region", "marital_status", "bmi_category",
    "smoking_status", "employment_status",
]

RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0,
}
MAX_RISK = 14  # heart disease (8) + diabetes or high blood pressure (6) — max seen in training
MIN_RISK = 0

PHYSICAL_ACTIVITY_RISK = {"High": 0, "Medium": 1, "Low": 4}
STRESS_RISK = {"High": 4, "Medium": 1, "Low": 0}
INSURANCE_PLAN_MAP = {"Bronze": 1, "Silver": 2, "Gold": 3}


def calculate_normalized_risk(medical_history: str) -> float:
    diseases = [d.strip().lower() for d in medical_history.split(" & ")]
    total = sum(RISK_SCORES.get(d, 0) for d in diseases)
    return (total - MIN_RISK) / (MAX_RISK - MIN_RISK)


def calculate_lifestyle_risk(physical_activity: str, stress_level: str) -> int:
    return PHYSICAL_ACTIVITY_RISK[physical_activity] + STRESS_RISK[stress_level]


def income_level_from_lakhs(income_lakhs: float) -> int:
    # Mirrors training map: '<10L':1, '10L - 25L':2, '25L - 40L':3, '> 40L':4
    if income_lakhs < 10:
        return 1
    elif income_lakhs <= 25:
        return 2
    elif income_lakhs <= 40:
        return 3
    return 4


def preprocess_input(input_dict: dict) -> pd.DataFrame:
    missing = [k for k in EXPECTED_KEYS if k not in input_dict]
    if missing:
        raise ValueError(f"Missing input keys: {missing}. Got: {list(input_dict.keys())}")

    # Start with all model features at 0
    row = {col: 0 for col in MODEL_FEATURES}

    # Numeric / ordinal features
    row["age"] = input_dict["age"]
    row["number_of_dependants"] = input_dict["number_of_dependants"]
    row["income_lakhs"] = input_dict["income_lakhs"]
    row["insurance_plan"] = INSURANCE_PLAN_MAP[input_dict["insurance_plan"]]

    # Engineered features
    row["normalized_risk_score"] = calculate_normalized_risk(input_dict["medical_history"])
    row["lifestyle_risk_score"] = calculate_lifestyle_risk(
        input_dict["physical_activity"], input_dict["stress_level"]
    )

    # One-hot features (drop_first=True baselines: Female, Northeast,
    # Married, Normal BMI, No Smoking, Freelancer)
    if input_dict["gender"] == "Male":
        row["gender_Male"] = 1
    if input_dict["region"] in ("Northwest", "Southeast", "Southwest"):
        row[f"region_{input_dict['region']}"] = 1
    if input_dict["marital_status"] == "Unmarried":
        row["marital_status_Unmarried"] = 1
    if input_dict["bmi_category"] in ("Obesity", "Overweight", "Underweight"):
        row[f"bmi_category_{input_dict['bmi_category']}"] = 1
    if input_dict["smoking_status"] in ("Occasional", "Regular"):
        row[f"smoking_status_{input_dict['smoking_status']}"] = 1
    if input_dict["employment_status"] in ("Salaried", "Self-Employed"):
        row[f"employment_status_{input_dict['employment_status']}"] = 1

    df = pd.DataFrame([row], columns=MODEL_FEATURES)

    # Scaler was fit WITH income_level present; add it, scale, then drop
    df["income_level"] = income_level_from_lakhs(input_dict["income_lakhs"])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df.drop("income_level", axis="columns")

    return df[MODEL_FEATURES]


def predict(input_dict: dict) -> float:
    input_df = preprocess_input(input_dict)
    prediction = model.predict(input_df)
    return round(float(prediction[0]), 2)

"""
main.py — Healthcare Premium Prediction Streamlit app
Run with:  streamlit run main.py
Input dict keys match prediction_helper.EXPECTED_KEYS exactly.
"""

import streamlit as st
from prediction_helper import predict

st.set_page_config(page_title="Health Insurance Premium Prediction", layout="wide")
st.title("Health Insurance Premium Prediction App")

# Category options — exact values from the training data
categorical_options = {
    "gender": ["Male", "Female"],
    "region": ["Northeast", "Northwest", "Southeast", "Southwest"],
    "marital_status": ["Married", "Unmarried"],
    "bmi_category": ["Normal", "Overweight", "Obesity", "Underweight"],
    "smoking_status": ["No Smoking", "Occasional", "Regular"],
    "employment_status": ["Salaried", "Self-Employed", "Freelancer"],
    "insurance_plan": ["Bronze", "Silver", "Gold"],
    "medical_history": [
        "No Disease", "Diabetes", "High blood pressure", "Heart disease",
        "Thyroid", "Diabetes & High blood pressure", "Diabetes & Heart disease",
        "Diabetes & Thyroid", "High blood pressure & Heart disease",
    ],
    "physical_activity": ["High", "Medium", "Low"],
    "stress_level": ["High", "Medium", "Low"],
}

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
with row1[1]:
    number_of_dependants = st.number_input("Number of Dependants", min_value=0, max_value=20, value=0, step=1)
with row1[2]:
    income_lakhs = st.number_input("Income in Lakhs", min_value=0, max_value=200, value=10, step=1)

with row2[0]:
    gender = st.selectbox("Gender", categorical_options["gender"])
with row2[1]:
    marital_status = st.selectbox("Marital Status", categorical_options["marital_status"])
with row2[2]:
    region = st.selectbox("Region", categorical_options["region"])

with row3[0]:
    bmi_category = st.selectbox("BMI Category", categorical_options["bmi_category"])
with row3[1]:
    smoking_status = st.selectbox("Smoking Status", categorical_options["smoking_status"])
with row3[2]:
    employment_status = st.selectbox("Employment Status", categorical_options["employment_status"])

with row4[0]:
    insurance_plan = st.selectbox("Insurance Plan", categorical_options["insurance_plan"])
with row4[1]:
    medical_history = st.selectbox("Medical History", categorical_options["medical_history"])
with row4[2]:
    physical_activity = st.selectbox("Physical Activity", categorical_options["physical_activity"])

stress_level = st.selectbox("Stress Level", categorical_options["stress_level"])

input_dict = {
    "age": age,
    "number_of_dependants": number_of_dependants,
    "income_lakhs": income_lakhs,
    "insurance_plan": insurance_plan,
    "medical_history": medical_history,
    "physical_activity": physical_activity,
    "stress_level": stress_level,
    "gender": gender,
    "region": region,
    "marital_status": marital_status,
    "bmi_category": bmi_category,
    "smoking_status": smoking_status,
    "employment_status": employment_status,
}

if st.button("Predict"):
    try:
        prediction = predict(input_dict)
        st.success(f"Predicted Annual Premium: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model("model/safe_model.json")   # Your model path
    return model

model = load_model()

# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    layout="centered"
)

st.markdown("""
<h2 style='text-align:center; color:#2c3e50;'>🏥 Diabetes 30-Day Readmission Predictor</h2>
<p style='text-align:center; font-size:17px;'>
Clinical decision support tool for discharge-time risk assessment.
</p>
""", unsafe_allow_html=True)

st.write("---")

# -------------------------------
# Clinical Form Inputs
# -------------------------------
st.subheader("Discharge Assessment Form")

with st.form("clinical_form"):
    age = st.slider("Patient Age", 18, 90, 60)

    race = st.selectbox("Race", ["Caucasian", "African American", "Asian", "Hispanic", "Other"])

    gender = st.selectbox("Gender", ["Male", "Female"])

    time_in_hospital = st.slider("Length of Stay (days)", 1, 30, 4)

    num_lab_procedures = st.slider("Number of Lab Procedures", 0, 100, 40)

    num_procedures = st.slider("Number of Procedures", 0, 15, 1)

    num_medications = st.slider("Number of Medications", 0, 40, 10)

    number_inpatient = st.slider("Prior Inpatient Visits (12 months)", 0, 10, 0)

    discharge_type = st.selectbox("Discharge Destination", [
        "Home", "Rehab", "Home Health", "SNF / Nursing Facility", "Other"
    ])

    med_change = st.selectbox("Medication Changed During Visit?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Risk")

# -------------------------------
# Mapping UI → Model Features
# -------------------------------

def encode_inputs():
    # Basic encodings (you’ll adjust based on your dataset)
    race_map = {"Caucasian":1, "African American":2, "Asian":3, "Hispanic":4, "Other":5}
    gender_map = {"Male":1, "Female":0}
    discharge_map = {"Home":1, "Rehab":2, "Home Health":3, "SNF / Nursing Facility":4, "Other":5}
    med_change_map = {"Yes":1, "No":0}

    return pd.DataFrame([{
        "age": age,
        "race": race_map[race],
        "gender": gender_map[gender],
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": 0,      # Hidden for now
        "number_inpatient": number_inpatient,
        "number_emergency": 0,        # Hidden for now
        "admission_type_id": 1,
        "discharge_disposition_id": discharge_map[discharge_type],
        "admission_source_id": 1,
        "diag_1": 1,
        "diag_2": 1,
        "diag_3": 1,
        "A1Cresult": 0,
        "diabetesMed": 1,
        "insulin": 1,
        "change": med_change_map[med_change],
        "had_prior_visit": 1 if number_inpatient > 0 else 0,
        "total_visits": number_inpatient,
        "procedure_per_day": num_procedures / max(time_in_hospital, 1),
        "age_group_numeric": age,
        "gender_race_combo": (gender_map[gender] * 10) + race_map[race]
    }])

# -------------------------------
# Model Prediction
# -------------------------------

if submitted:
    X = encode_inputs()
    dmatrix = xgb.DMatrix(X)

    prob = model.predict(dmatrix)[0]
    risk_index = int(prob * 100)

    # Color classification
    if risk_index <= 30:
        color = "green"
        label = "LOW RISK"
        recommendation = "Standard discharge follow-up recommended."
    elif risk_index <= 60:
        color = "orange"
        label = "MODERATE RISK"
        recommendation = "Consider scheduling early follow-up within 1 week."
    else:
        color = "red"
        label = "HIGH RISK"
        recommendation = "Recommend high-intensity follow-up and care coordination."

    # Display results
    st.markdown(f"""
        <div style='text-align:center; padding:20px; border-radius:10px; background-color:#f7f9fa;'>
            <h3 style='color:{color};'>Risk Index: {risk_index}</h3>
            <h4 style='color:{color};'>{label}</h4>
            <p style='font-size:16px;'>{recommendation}</p>
        </div>
    """, unsafe_allow_html=True)

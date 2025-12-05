import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model(): 
    model = xgb.Booster()
    model.load_model("model/safe_model.json")  # Path inside repo
    return model

model = load_model()

# -------------------------
# SHAP Explainer
# -------------------------
explainer = shap.TreeExplainer(model)

# -------------------------
# UI CONFIG
# -------------------------
st.set_page_config(page_title="Diabetes Readmission Predictor",
                   layout="centered")

st.markdown("""
    <h2 style='text-align:center; color:#2c3e50;'>🩺 Diabetes 30-Day Readmission Predictor</h2>
    <p style='text-align:center; font-size:17px;'>
        Clinical decision support tool for discharge-time risk assessment.
    </p>
""", unsafe_allow_html=True)

# -------------------------
# FRONT-END FORM (10 CLINICAL QUESTIONS)
# -------------------------
with st.form("patient_form"):
    st.subheader("Patient Clinical Information")

    age = st.selectbox("Age Group", 
                       ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])

    time_in_hospital = st.slider("Length of Stay (days)", 1, 14, 4)

    a1c = st.selectbox("Recent A1C Result", ["Norm", ">7", ">8"])

    primary_dx = st.number_input("Primary Diagnosis Code (ICD-9)", 100, 999, 250)

    num_meds = st.slider("Number of Medications", 1, 30, 10)

    num_procedures = st.slider("Number of Procedures", 0, 6, 1)

    prior_inpatient = st.slider("Prior Inpatient Visits", 0, 10, 0)

    discharge = st.selectbox("Discharge Disposition",
                             ["Home", "Rehab", "Skilled Nursing", "Other"])

    medication_change = st.selectbox("Medication Change During Visit?", ["Yes", "No"])

    insulin_status = st.selectbox("Current Insulin Use", ["No", "Steady", "Up", "Down"])

    submitted = st.form_submit_button("Predict Readmission Risk")

# -------------------------
# MAPPING → MODEL FEATURES
# -------------------------
def map_inputs_to_features():
    age_map = {
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
        '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }

    discharge_map = {
        "Home": 1, "Rehab": 3, "Skilled Nursing": 5, "Other": 6
    }

    a1c_map = {"Norm": 0, ">7": 1, ">8": 2}

    insulin_map = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}

    change_map = {"Yes": 1, "No": 0}

    # Build single-row dataframe for model
    row = pd.DataFrame([{
        "age": age,
        "race": 0,
        "gender": 0,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": 40,
        "num_procedures": num_procedures,
        "num_medications": num_meds,
        "number_outpatient": 0,
        "number_inpatient": prior_inpatient,
        "number_emergency": 0,
        "admission_type_id": 1,
        "discharge_disposition_id": discharge_map[discharge],
        "admission_source_id": 1,
        "diag_1": primary_dx,
        "diag_2": 250,
        "diag_3": 250,
        "A1Cresult": a1c_map[a1c],
        "diabetesMed": 1,
        "insulin": insulin_map[insulin_status],
        "change": change_map[medication_change],
        "had_prior_visit": 1 if prior_inpatient > 0 else 0,
        "total_visits": prior_inpatient,
        "procedure_per_day": num_procedures / max(time_in_hospital, 1),
        "age_group_numeric": age_map[age],
        "gender_race_combo": 0
    }])

    return row

# -------------------------
# RUN PREDICTION
# -------------------------
if submitted:
    X = map_inputs_to_features()

    dmatrix = xgb.DMatrix(X)
    prob = float(model.predict(dmatrix)[0])

    risk_index = int(prob * 100)

    # -------------------------
    # COLOR-CODED RISK OUTPUT
    # -------------------------
    if risk_index < 30:
        risk_level = "LOW"
        color = "green"
        advice = "Continue routine follow-up and outpatient care."
    elif risk_index < 60:
        risk_level = "MEDIUM"
        color = "orange"
        advice = "Consider enhanced discharge planning and close follow-up."
    else:
        risk_level = "HIGH"
        color = "red"
        advice = "Recommend intensive transitional care and early follow-up within 7 days."

    st.markdown(f"""
        <div style="padding:20px; border-radius:10px; background-color:#f7f9fc;">
            <h3 style="color:{color}; text-align:center;">
                Risk Index: {risk_index} / 100<br>
                ({risk_level} Risk)
            </h3>
            <p style="text-align:center; font-size:18px;">{advice}</p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # ADVANCED: SHAP DETAILS
    # -------------------------
    with st.expander("🔍 Advanced: Feature Impact (SHAP Values)"):
        shap_values = explainer.shap_values(X)

        st.write("Top contributing factors for this patient:")

        fig, ax = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig)

        st.caption("These values show how each feature influenced the model's prediction.")

# End of app.py

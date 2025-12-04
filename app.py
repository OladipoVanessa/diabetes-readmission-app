import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

st.title("🩺 Diabetes Readmission Predictor")
st.write("Upload a patient's data below to predict 30-day readmission risk.")

# Load model once
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("safe_model.json")  # load directly from file
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload patient data CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    prediction = model.predict(data)
    st.write("### Prediction:")
    st.write(prediction)

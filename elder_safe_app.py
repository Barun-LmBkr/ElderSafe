import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd

# Load the trained model and scaler
model = joblib.load('xgboost_model.joblib')
scaler = joblib.load('scaler.joblib')

# Function to predict fall risk
def predict_fall_risk(data):
    # Scale the data
    data_scaled = scaler.transform([data])
    # Predict using the model
    prediction = model.predict(data_scaled)
    return prediction[0]

# Function to explain prediction with SHAP
def explain_prediction(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.summary_plot(shap_values, data)

# App title
st.title('ElderSafe - Fall Risk Prediction')

# Input fields for the user
age = st.number_input('Age', min_value=60, max_value=120, step=1)
walking_speed = st.number_input('Walking Speed (m/s)', min_value=0.0, max_value=10.0, step=0.1)
missed_meals = st.number_input('Number of Missed Meals', min_value=0, max_value=10, step=1)
sleep_hours = st.number_input('Sleep Hours', min_value=1, max_value=12, step=1)
medication_adherence = st.selectbox('Medication Adherence', ['Yes', 'No'])
past_falls = st.selectbox('Past Falls', ['Yes', 'No'])
mobility_aid = st.selectbox('Mobility Aid', ['Yes', 'No'])

# Map categorical features to numeric
medication_adherence = 1 if medication_adherence == 'Yes' else 0
past_falls = 1 if past_falls == 'Yes' else 0
mobility_aid = 1 if mobility_aid == 'Yes' else 0

# Prepare the input data
input_data = [age, walking_speed, missed_meals, sleep_hours, medication_adherence, past_falls, mobility_aid]

# Button to predict fall risk
if st.button('Predict Fall Risk'):
    risk = predict_fall_risk(input_data)
    risk_label = 'High Risk' if risk == 1 else 'Low Risk'
    st.write(f'Predicted Fall Risk: {risk_label}')

    # Explain the prediction with SHAP
    st.subheader('Explanation of Prediction:')
    explain_prediction(np.array(input_data).reshape(1, -1))


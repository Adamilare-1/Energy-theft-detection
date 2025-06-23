# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 22:24:37 2025

@author: Adamilare
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load("best_model.pkl")  
model_name = "K-Nearest Neighbors "
model_accuracy = 0.975                   


st.set_page_config(page_title="Energy Theft Detection",
                   page_icon="⚡",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.sidebar.header("👋 Model Info & Guide")
st.sidebar.markdown(f"**Model Name:** {model_name}")
st.sidebar.markdown(f"**Validation Accuracy:** {model_accuracy * 100:.2f}%")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Input Definitions:**
- **Reported Usage**: Usage measured by the official meter.
- **Actual Usage**: Usage measured by a smart meter.
- **Voltage Fluctuations**: Abnormal fluctuations per month.
- **Average Daily Load**: Typical daily usage.
- **Number of Blackouts**: Frequency of outages per month.

**Instructions:**  
Enter the required values for the month and click **Predict Theft**.
""")

st.title("⚡ Energy Theft Detection System")
st.markdown("""
This tool predicts the likelihood of energy theft by comparing **reported usage** from the Energy provider versus **actual usage** measured by a smart meter.

**Instructions**: Enter the values for the month and click **Predict Theft**.
""")
st.markdown("---")

reported_usage = st.number_input("Reported Usage (kWh) – official meter:", value=250.0, min_value=0.0, step=1.0)
actual_usage = st.number_input("Actual Usage (kWh) – smart meter:", value=300.0, min_value=0.0, step=1.0)
voltage_fluctuations = st.number_input("Voltage Fluctuations (count):", value=3, min_value=0, step=1)
avg_daily_load = st.number_input("Average Daily Load (kWh):", value=10.0, min_value=0.0, step=0.1)
no_of_blackouts = st.number_input("Number of Blackouts:", value=1, min_value=0, step=1)

if st.button("🔍 Predict Theft"):
    input_data = pd.DataFrame([[reported_usage, actual_usage, voltage_fluctuations, avg_daily_load, no_of_blackouts]],
                              columns=["reported_usage", "actual_usage", "voltage_fluctuations", "avg_daily_load", "no_of_blackouts"])
    prediction = model.predict(input_data)[0]
    prediction_prob = model.predict_proba(input_data)[0]

    if prediction == 1:
        theft_confidence = prediction_prob[1] * 100
        st.error(f"❌ Energy Theft Detected! (Confidence: {theft_confidence:.2f}%)")
    else:
        no_theft_confidence = prediction_prob[0] * 100
        st.success(f"✅ No Energy Theft Detected. (Confidence: {no_theft_confidence:.2f}%)")

    # Visualisation of Confidence
    st.progress(int(theft_confidence if prediction == 1 else no_theft_confidence))
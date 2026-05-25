# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD FILES
# =========================

model = joblib.load("svm_model.pkl")

scaler = joblib.load("scaler.pkl")

# =========================
# STREAMLIT PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("🩺 Breast Cancer Prediction using SVM")

st.write("Enter Medical Values Below")

# =========================
# INPUT FIELDS
# =========================

col1, col2 = st.columns(2)

with col1:

    radius_mean = st.number_input(
        "Radius Mean",
        value=14.0
    )

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        value=90.0
    )

    smoothness_mean = st.number_input(
        "Smoothness Mean",
        value=0.1
    )

    concavity_mean = st.number_input(
        "Concavity Mean",
        value=0.1
    )

with col2:

    texture_mean = st.number_input(
        "Texture Mean",
        value=20.0
    )

    area_mean = st.number_input(
        "Area Mean",
        value=600.0
    )

    compactness_mean = st.number_input(
        "Compactness Mean",
        value=0.1
    )

    symmetry_mean = st.number_input(
        "Symmetry Mean",
        value=0.2
    )

# =========================
# CREATE INPUT ARRAY
# =========================

input_data = np.array([
    [
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        symmetry_mean
    ]
])

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)

    confidence = np.max(probability) * 100

    # Output
    if prediction[0] == 1:

        st.error(
            f"Prediction: Malignant Cancer\n\nConfidence: {confidence:.2f}%"
        )

    else:

        st.success(
            f"Prediction: Benign Cancer\n\nConfidence: {confidence:.2f}%"
        )
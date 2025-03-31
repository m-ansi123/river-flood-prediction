import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("flood_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🌊 River Flood Prediction System")
st.markdown("Enter environmental parameters to predict flood risk.")

# Input fields
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
river_level = st.number_input("River Level (m)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, step=0.1)

# Prediction
if st.button("Predict Flood Risk"):
    features = np.array([[rainfall, river_level, temperature]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    # Display prediction result
    risk_levels = {0: "Low Risk (🟢)", 1: "Medium Risk (🟡)", 2: "High Risk (🔴)"}
    st.subheader(f"Flood Risk Level: {risk_levels[prediction]}")

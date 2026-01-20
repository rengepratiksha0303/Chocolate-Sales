import streamlit as st
import pickle
import numpy as np
from datetime import date

# Page config
st.set_page_config(page_title="Chocolate Sales Prediction", layout="centered")

st.title("🍫 Chocolate Sales Prediction App")
st.write("Predict **Boxes Shipped** using a trained ML model")

# Load model
with open("chocolate_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Date input
selected_date = st.date_input(
    "Select Date",
    value=date.today()
)

# Extract features
year = selected_date.year
month = selected_date.month
day = selected_date.day

# Predict button
if st.button("Predict Boxes Shipped"):
    input_data = np.array([[year, month, day]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"📦 Predicted Boxes Shipped: **{int(prediction[0])}**")

import streamlit as st
import pickle
import numpy as np

# ✅ Load everything
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("poly.pkl", "rb") as f:
    poly = pickle.load(f)

st.title("🏠 House Price Prediction")

total_rooms = st.number_input("Enter total rooms:", min_value=0.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[total_rooms]])
    input_poly = poly.transform(input_data)
    scaled_pred = model.predict(input_poly)
    pred = scaler.inverse_transform(scaled_pred)

    st.success(f"Predicted House Price: ${pred[0][0]:,.2f}")

import streamlit as st
import joblib
import numpy as np

model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🐔 Poultry Mortality Prediction System")

st.write("Predict next-day poultry mortality risk using XGBoost.")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
co2 = st.number_input("CO2")
flock_age = st.number_input("Flock Age")
prev_mortality_rate = st.number_input("Previous Mortality Rate")

if st.button("Predict"):

    data = np.array([[
        temperature,
        humidity,
        co2,
        flock_age,
        prev_mortality_rate
    ]])

    data = scaler.transform(data)

    probability = model.predict_proba(data)[0][1]

    if probability < 0.3:
        st.success(f"🟢 LOW RISK ({probability*100:.2f}%)")

    elif probability < 0.7:
        st.warning(f"🟡 MEDIUM RISK ({probability*100:.2f}%)")

    else:
        st.error(f"🔴 HIGH RISK ({probability*100:.2f}%)")

st.subheader("Feature Importance")
st.image("static/feature_importance.png")

st.subheader("Confusion Matrix")
st.image("static/confusion_matrix.png")

st.subheader("Model Accuracy Comparison")
st.image("static/accuracy_comparison.png")

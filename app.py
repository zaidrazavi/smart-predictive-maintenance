import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Smart Predictive Maintenance System")

st.write("Predict machine failure using sensor values")

air_temp = st.slider("Air Temperature (K)", 290, 320)
process_temp = st.slider("Process Temperature (K)", 300, 340)
rpm = st.slider("Rotational Speed (rpm)", 1000, 3000)
torque = st.slider("Torque (Nm)", 0, 100)
tool_wear = st.slider("Tool Wear (min)", 0, 300)

if st.button("Predict"):
    data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Machine Failure Likely!")
    else:
        st.success("✅ Machine is Safe")
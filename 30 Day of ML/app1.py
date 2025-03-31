import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability.")

# User input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.radio("Gender", ["Male", "Female"])
monthly_spending = st.number_input("Monthly Spending ($)", min_value=0.0, max_value=500.0, value=50.0)
contract_length = st.slider("Contract Length (months)", min_value=1, max_value=24, value=12)
num_support_tickets = st.number_input("Number of Support Tickets", min_value=0, max_value=10, value=1)

# Convert gender to numerical format
gender = 1 if gender == "Male" else 0

# Make prediction
if st.button("Predict Churn"):
    user_data = [[age, gender, monthly_spending, contract_length, num_support_tickets]]
    user_data_scaled = scaler.transform(user_data)
    
    churn_prediction = model.predict(user_data_scaled)[0]
    churn_probability = model.predict_proba(user_data_scaled)[0][1]
    
    if churn_prediction == 1:
        st.error(f"🚨 This customer is **likely to churn** (Probability: {churn_probability:.2f})")
    else:
        st.success(f"✅ This customer is **not likely to churn** (Probability: {1 - churn_probability:.2f})")

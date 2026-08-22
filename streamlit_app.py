import requests
import streamlit as st
import joblib
import numpy as np
import os
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_models():
    model_path = hf_hub_download(
        repo_id="PranjulKushwaha/bank-loan-default-prediction",
        filename="best_model.pkl",
        repo_type="model"
    )
    scaler_path = hf_hub_download(
        repo_id="PranjulKushwaha/bank-loan-default-prediction",
        filename="scaler.pkl",
        repo_type="model"
    )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_models()

st.title("🏦 Bank Loan Default Prediction")
st.markdown("### Customer Risk Assessment System")

st.sidebar.header("Customer Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=0, value=5000)
debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 1.0, 0.3)
revolving_util = st.sidebar.slider("Revolving Utilization", 0.0, 1.0, 0.5)
open_credit_lines = st.sidebar.number_input("Open Credit Lines", min_value=0, value=5)
real_estate_loans = st.sidebar.number_input("Real Estate Loans", min_value=0, value=1)
dependents = st.sidebar.number_input("Number of Dependents", min_value=0, value=1)
late_30_59 = st.sidebar.number_input("Late 30-59 Days", min_value=0, value=0)
late_60_89 = st.sidebar.number_input("Late 60-89 Days", min_value=0, value=0)
late_90 = st.sidebar.number_input("Late 90+ Days", min_value=0, value=0)

if st.button("🔍 Predict Risk"):
    monthly_debt = monthly_income * debt_ratio
    total_late = late_30_59 + late_60_89 + late_90
    credit_risk = revolving_util * open_credit_lines
    income_per_dependent = monthly_income / (dependents + 1)

    input_data = [[
        revolving_util, age, late_30_59, debt_ratio,
        monthly_income, open_credit_lines, late_90,
        real_estate_loans, late_60_89, dependents,
        monthly_debt, total_late, credit_risk, income_per_dependent
    ]]

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if probability < 0.30:
        st.success(f"✅ LOW RISK — Probability: {probability:.2%}")
    elif probability < 0.60:
        st.warning(f"⚠️ MEDIUM RISK — Probability: {probability:.2%}")
    else:
        st.error(f"🚨 HIGH RISK — Probability: {probability:.2%}")
#------------------------------------------------------------------------------

st.divider()
st.subheader("🤖 Loan Policy Chatbot")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    response = requests.post(
        "http://localhost:8000/auth/token",
        data={"username":username, "password":password}
    )
    if response.status_code == 200:
        st.session_state["token"] = response.json()["access_token"]
        st.success("✅ Login successful!")
    else:
        st.error("❌ Wrong username or password")

if "token" in st.session_state:
    question = st.text_input("Apna sawaal likho")
    if st.button("Ask"):
        response = requests.post(
            "http://localhost:8000/v1/ask",
            json={"question":question},
            headers={"Authorization": f"Bearer {st.session_state['token']}"}
        )
        st.write(response.json()["answer"])
import streamlit as st
import pandas as pd
import joblib

# =======================
# Load ML Model
# =======================
MODEL_PATH = "model/churn_model.pkl"
model = joblib.load(MODEL_PATH)

# =======================
# Streamlit UI Styling
# =======================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .header {
        font-size: 32px;
        font-weight: 700;
        color: #2A4365;
        text-align: center;
        padding-bottom: 10px;
    }
    .subheader {
        font-size: 18px;
        color: #4A5568;
        text-align: center;
        padding-bottom: 25px;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        padding-top: 50px;
        color: gray;
    }
    .result-box {
        border-radius: 12px;
        padding: 20px;
        color: white;
        font-size: 22px;
        font-weight: bold;
    }
    .success {
        background: #38A169;
    }
    .danger {
        background: #E53E3E;
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# Header
# =======================
st.markdown("<div class='header'>📊 Customer Churn Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>An intelligent ML-powered tool that predicts which telecom customers are likely to leave</div>", unsafe_allow_html=True)

st.write("---")

# =======================
# Sidebar Input
# =======================
st.sidebar.title("🔍 Input Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

# =======================
# Data Convert
# =======================
input_data = pd.DataFrame({
    "gender": [1 if gender == "Female" else 0],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [1 if Partner == "Yes" else 0],
    "Dependents": [1 if Dependents == "Yes" else 0],
    "tenure": [tenure],
    "PhoneService": [1 if PhoneService == "Yes" else 0],
    "PaperlessBilling": [1 if PaperlessBilling == "Yes" else 0],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
    "Contract_One year": [1 if Contract == "One year" else 0],
    "Contract_Two year": [1 if Contract == "Two year" else 0],
    "InternetService_Fiber optic": [1 if InternetService == "Fiber optic" else 0],
    "InternetService_No": [1 if InternetService == "No" else 0],
    "OnlineSecurity_Yes": [1 if OnlineSecurity == "Yes" else 0],
    "TechSupport_Yes": [1 if TechSupport == "Yes" else 0],
})

# =======================
# Predict
# =======================
if st.button("🚀 Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("### 🧠 Prediction Result")
    if prediction == 1:
        st.markdown(
            f"<div class='result-box danger'>⚠️ High Churn Risk — Probability: {probability:.2f}</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='result-box success'>✅ Customer Likely to Stay — Probability: {probability:.2f}</div>",
            unsafe_allow_html=True)

# =======================
# Footer
# =======================
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & Machine Learning | Ideal for Portfolio Projects</div>", unsafe_allow_html=True)

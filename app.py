import streamlit as st
import joblib
import numpy as np
import pandas as pd
from fpdf import FPDF
import time

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# --------------------- MODEL LOAD ---------------------
@st.cache_resource
def load_model():
    return joblib.load("model/churn_model.pkl")

model = load_model()

# --------------------- SIDEBAR INPUT ---------------------
st.sidebar.header("🔍 Input Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# --------------------- MODEL INPUT PREPARATION ---------------------
def preprocess_data():
    row = pd.DataFrame([{
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": senior,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0
    }])
    return row

# --------------------- PDF REPORT ---------------------
def generate_pdf(pred, prob):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Customer Churn Prediction Report", ln=1, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {'Customer WILL Churn' if pred == 1 else 'Customer will NOT Churn'}", ln=1)
    pdf.cell(200, 10, txt=f"Churn Probability: {prob:.2%}", ln=1)
    file_path = "churn_report.pdf"
    pdf.output(file_path)
    return file_path

# --------------------- MAIN UI ---------------------
st.title("📈 Customer Churn Prediction")
st.write("An intelligent ML-powered tool that predicts telecom customer churn based on user details.")

predict_btn = st.button("🚀 Predict Churn", use_container_width=True)

# --------------------- PREDICTION ---------------------
if predict_btn:
    st.write("⏳ Processing input…")
    with st.spinner("Running prediction..."):
        time.sleep(1.2)
        input_df = preprocess_data()
        pred = model.predict(input_df)[0]

        # Probability (only if supported)
        try:
            prob = model.predict_proba(input_df)[0][1]
        except:
            prob = 1.0 if pred == 1 else 0.0

    st.success("🎯 Prediction Complete!")

    # ---- Probability Gauge ----
    st.subheader("📊 Churn Probability")
    st.progress(int(prob * 100))

    # ---- Result ----
    if pred == 1:
        st.error("🚨 High Risk: Customer is LIKELY to churn!")
    else:
        st.success("🟢 Safe: Customer is unlikely to churn.")

    # ---- PDF Download ----
    file = generate_pdf(pred, prob)
    with open(file, "rb") as f:
        st.download_button("📥 Download Report as PDF", data=f, file_name="Churn_Result.pdf", mime="application/pdf")

# --------------------- FOOTER ---------------------
st.markdown("""
<hr>
<div style="text-align:center; font-size:14px;">
Built by <b>Raj Lalji Pandey</b> using Streamlit & Machine Learning<br>
<a href="https://github.com/rajlaljipandey" target="_blank">GitHub</a> |
<a href="https://www.linkedin.com/in/raj-pandey-51288a237/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

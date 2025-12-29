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
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Customer Churn Prediction Report", ln=1, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
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
    with st.spinner("⏳ Processing input..."):
        time.sleep(1.3)

    input_df = pd.DataFrame([[
        gender, senior, partner, dependents, tenure, phone, paperless
    ]], columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling'])

    # Encode
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'Yes':1, 'No':0})

    input_df['gender'] = input_df['gender'].map({'Male':1,'Female':0})

    # 🧠 FIX — Match model training columns
    try:
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_cols]
    except:
        pass

    input_array = input_df.values

    pred = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0][1]

    if pred == 1:
        st.error("🔴 High Risk: Customer likely to churn")
    else:
        st.success("🟢 Safe: Customer unlikely to churn")

    st.write("📊 Churn Probability:")
    st.progress(int(prob * 100))

    file = generate_pdf(pred, prob)
    with open(file, "rb") as f:
        st.download_button("📥 Download PDF", data=f, file_name="Churn_Result.pdf", mime="application/pdf")

# --------------------- FOOTER ---------------------
st.markdown("""
<hr>
<div style="text-align:center; font-size:14px;">
Built by <b>Raj Lalji Pandey</b> using Streamlit & Machine Learning<br>
<a href="https://github.com/rajlaljipandey" target="_blank">GitHub</a> |
<a href="https://www.linkedin.com/in/raj-pandey-51288a237/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/churn_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(to right, #3b82f6, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer {
        width: 100%;
        text-align: center;
        padding: 12px;
        font-size: 14px;
        color: #6b7280;
        border-top: 1px solid #e5e7eb;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Input UI
st.sidebar.header("🔍 Input Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", 1, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# Convert inputs
def encode(val):
    return 1 if val == "Yes" else 0

input_data = np.array([[1 if gender == "Male" else 0,
                        senior,
                        encode(partner),
                        encode(dependents),
                        tenure,
                        encode(phone),
                        encode(paperless)]])

# ---- Main Layout ----
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<h1 class='main-title'>📱 Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.write("An intelligent ML-powered tool that predicts which telecom customers are likely to leave.")


# Button
predict_btn = st.button("🚀 Predict Churn", use_container_width=True)

# Collect user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, step=1)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

# Raw input list stored here 👇
input_data = [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, PaperlessBilling]

# Display Prediction
# ------------------------------------
if predict_btn:
    # ------------ PREPROCESS USER INPUT ------------
    input_df = pd.DataFrame([input_data], columns=[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'PaperlessBilling'
    ])

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})

    # Final processed data
    input_data = input_df

    # ------------ Prediction ------------
    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("🚨 High Risk: Customer is likely to churn", icon="⚠️")
    else:
        st.success("🟢 Safe: Customer is unlikely to churn", icon="😊")


# ------------------------------------
# Footer
# ------------------------------------
st.markdown("""
<div class='footer'>
    Built by <b>Raj Lalji Pandey</b> using Streamlit & Machine Learning |
    <a href='https://github.com/rajlaljipandey' target='_blank'>GitHub</a> ·
    <a href='https://www.linkedin.com/in/raj-pandey-51288a237/' target='_blank'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)

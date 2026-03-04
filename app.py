import streamlit as st
import joblib
import pandas as pd
import numpy as np
# This is vedant deshmukh
# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")
st.title("🏦 Credit Loan Approval Predictor")
st.markdown("Enter applicant details below to check loan eligibility.")
st.markdown("---")

# ----------------------------
# Load Model and Scaler
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Financial Information")

    applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=1.0, value=10000.0)
    coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000.0, value=20000.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", 0.1, 0.6, 0.3)
    savings = st.number_input("Total Savings ($)", min_value=0.0, value=5000.0)
    collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=15000.0)

with col2:
    st.header("👤 Applicant Profile")

    age = st.slider("Age", 21, 60, 30)
    dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
    credit_score = st.slider("Credit Score", 550, 800, 680)
    existing_loans = st.number_input("Number of Existing Loans", min_value=0, value=0)

    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    loan_purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal"])
    employer_cat = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])

# ----------------------------
# Prediction
# ----------------------------
if st.button("🚀 Predict Eligibility", use_container_width=True):

    # Feature Engineering (must match training)
    DTI_Ratio_sq = dti_ratio ** 2
    Credit_Score_sq = credit_score ** 2
    Applicant_Income_log = np.log(applicant_income)

    # Create dictionary of ALL 27 features
    data = {
        'Coapplicant_Income': coapplicant_income,
        'Age': age,
        'Dependents': dependents,
        'Existing_Loans': existing_loans,
        'Savings': savings,
        'Collateral_Value': collateral_value,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,

        'Education_Level': 1 if education_level == "Graduate" else 0,

        'Employment_Status_Salaried': 1 if employment_status == "Salaried" else 0,
        'Employment_Status_Self-employed': 1 if employment_status == "Self-employed" else 0,
        'Employment_Status_Unemployed': 1 if employment_status == "Unemployed" else 0,

        'Marital_Status_Single': 1 if marital_status == "Single" else 0,

        'Loan_Purpose_Car': 1 if loan_purpose == "Car" else 0,
        'Loan_Purpose_Education': 1 if loan_purpose == "Education" else 0,
        'Loan_Purpose_Home': 1 if loan_purpose == "Home" else 0,
        'Loan_Purpose_Personal': 1 if loan_purpose == "Personal" else 0,

        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0,

        'Gender_Male': 1 if gender == "Male" else 0,

        'Employer_Category_Government': 1 if employer_cat == "Government" else 0,
        'Employer_Category_MNC': 1 if employer_cat == "MNC" else 0,
        'Employer_Category_Private': 1 if employer_cat == "Private" else 0,
        'Employer_Category_Unemployed': 1 if employer_cat == "Unemployed" else 0,

        'DTI_Ratio_sq': DTI_Ratio_sq,
        'Credit_Score_sq': Credit_Score_sq,
        'Applicant_Income_log': Applicant_Income_log
    }

    df = pd.DataFrame([data])

    # Ensure same column order as training
    FEATURES = [
    'Coapplicant_Income',
    'Age',
    'Dependents',
    'Existing_Loans',
    'Savings',
    'Collateral_Value',
    'Loan_Amount',
    'Loan_Term',
    'Education_Level',
    'Employment_Status_Salaried',
    'Employment_Status_Self-employed',
    'Employment_Status_Unemployed',
    'Marital_Status_Single',
    'Loan_Purpose_Car',
    'Loan_Purpose_Education',
    'Loan_Purpose_Home',
    'Loan_Purpose_Personal',
    'Property_Area_Semiurban',
    'Property_Area_Urban',
    'Gender_Male',
    'Employer_Category_Government',
    'Employer_Category_MNC',
    'Employer_Category_Private',
    'Employer_Category_Unemployed',
    'DTI_Ratio_sq',
    'Credit_Score_sq',
    'Applicant_Income_log'
]

    df = df.reindex(columns=FEATURES, fill_value=0)

    # 🔥 SCALE INPUT BEFORE PREDICTION
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    st.markdown("---")
    st.write("Approval Probability:", round(probability, 3))

    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {probability:.2%})")
        st.balloons()
    else:
        st.error(f"❌ Loan Rejected. (Confidence: {(1 - probability):.2%})")
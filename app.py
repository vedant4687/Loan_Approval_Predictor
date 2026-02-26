
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Loan Predictor", layout="wide")
st.title("🏦 Credit Loan Approval Predictor")
st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load('loan_model.pkl')

# INPUT FORM - UPDATE THESE to match YOUR features exactly
col1, col2 = st.columns(2)
with col1:
    st.header("📊 Financial Details")
    income = st.number_input("Annual Income (₹)", 0, 10000000, 500000)
    credit_score = st.slider("Credit Score", 300, 900, 700)
    
with col2:
    st.header("👤 Personal Details")
    age = st.slider("Age", 18, 70, 30)
    loan_amount = st.number_input("Loan Amount (₹)", 10000, 5000000, 500000)

employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed"])
education = st.selectbox("Highest Education", ["High School", "Graduate", "Post Graduate"])

if st.button("🚀 Predict Eligibility", type="primary", use_container_width=True):
    features = {
        'income': income, 'age': age, 'credit_score': credit_score,
        'loan_amount': loan_amount, 'employment_type': employment_type,
        'education': education
    }
    
    df = pd.DataFrame([features])
    model = load_model()
    
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    st.markdown("### 📈 Prediction Results")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if prediction == 1:
            st.success("✅ **LOAN APPROVED**")
        else:
            st.error("❌ **LOAN REJECTED**")
    
    with col2:
        st.metric("Approval Probability", f"{probability:.1%}")
    
    st.markdown("---")
    st.caption("Built with ❤️ using your sklearn model")

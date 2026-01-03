import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Setup Page
st.set_page_config(page_title="Retention Savior", page_icon="🛡️", layout="wide")

# 2. Load Assets
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'churn_xgb.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    shap_img_path = os.path.join(base_dir, 'app', 'shap_summary.png')
    
    return joblib.load(model_path), joblib.load(scaler_path), shap_img_path

try:
    model, scaler, shap_img_path = load_assets()
except FileNotFoundError:
    st.error("❌ Assets not found. Run src/train.py and src/explain.py first!")
    st.stop()

# 3. Sidebar: Inputs
st.sidebar.header("👤 Customer Profile")

# Demographics
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.checkbox("Senior Citizen")
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

# Services
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 24)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No internet service", "No", "Yes"])
online_backup = st.sidebar.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device_protection = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No internet service", "No", "Yes"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

# Contract & Billing
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.radio("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 1500.0)

# 4. Preprocessing Engine
def preprocess_input():
    # The Exact Column Order from Training (CRITICAL)
    cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 
            'MonthlyCharges', 'TotalCharges', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
            'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 
            'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
            'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 
            'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 
            'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 
            'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
            'PaymentMethod_Mailed check']

    # Create empty row
    input_df = pd.DataFrame(0, index=[0], columns=cols)
    
    # Map Binary & Numerical
    input_df['gender'] = 1 if gender == "Female" else 0
    input_df['SeniorCitizen'] = 1 if senior else 0
    input_df['Partner'] = 1 if partner == "Yes" else 0
    input_df['Dependents'] = 1 if dependents == "Yes" else 0
    input_df['PhoneService'] = 1 if phone_service == "Yes" else 0
    input_df['PaperlessBilling'] = 1 if paperless == "Yes" else 0
    
    # Scale Numerical
    scaled_nums = scaler.transform([[tenure, monthly_charges, total_charges]])
    input_df['tenure'] = scaled_nums[0][0]
    input_df['MonthlyCharges'] = scaled_nums[0][1]
    input_df['TotalCharges'] = scaled_nums[0][2]

    # Map One-Hot Columns
    if multiple_lines == "No phone service": input_df['MultipleLines_No phone service'] = 1
    if multiple_lines == "Yes": input_df['MultipleLines_Yes'] = 1
    
    if internet_service == "Fiber optic": input_df['InternetService_Fiber optic'] = 1
    if internet_service == "No": input_df['InternetService_No'] = 1
    
    if online_security == "No internet service": input_df['OnlineSecurity_No internet service'] = 1
    if online_security == "Yes": input_df['OnlineSecurity_Yes'] = 1

    if online_backup == "No internet service": input_df['OnlineBackup_No internet service'] = 1
    if online_backup == "Yes": input_df['OnlineBackup_Yes'] = 1

    if device_protection == "No internet service": input_df['DeviceProtection_No internet service'] = 1
    if device_protection == "Yes": input_df['DeviceProtection_Yes'] = 1

    if tech_support == "No internet service": input_df['TechSupport_No internet service'] = 1
    if tech_support == "Yes": input_df['TechSupport_Yes'] = 1

    if streaming_tv == "No internet service": input_df['StreamingTV_No internet service'] = 1
    if streaming_tv == "Yes": input_df['StreamingTV_Yes'] = 1
    
    if streaming_movies == "No internet service": input_df['StreamingMovies_No internet service'] = 1
    if streaming_movies == "Yes": input_df['StreamingMovies_Yes'] = 1
    
    if contract == "One year": input_df['Contract_One year'] = 1
    if contract == "Two year": input_df['Contract_Two year'] = 1
    
    if payment_method == "Credit card (automatic)": input_df['PaymentMethod_Credit card (automatic)'] = 1
    if payment_method == "Electronic check": input_df['PaymentMethod_Electronic check'] = 1
    if payment_method == "Mailed check": input_df['PaymentMethod_Mailed check'] = 1

    return input_df

# 5. Dashboard Layout
st.title("🛡️ AjayDataLabs | Retention Savior")
st.markdown("### Real-time Customer Churn Prediction")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔮 Risk Analysis")
    input_data = preprocess_input()
    
    if st.button("Predict Churn Risk"):
        prob = model.predict_proba(input_data)[0][1]
        
        # Dynamic Gauge
        st.metric(label="Churn Probability", value=f"{prob:.1%}")
        
        if prob > 0.6:
            st.error("🚨 HIGH RISK CUSTOMER")
            st.info("💡 Recommendation: Immediate intervention required. Offer 12-month contract discount.")
        elif prob > 0.3:
            st.warning("⚠️ MODERATE RISK")
            st.info("💡 Recommendation: Monitor usage frequency.")
        else:
            st.success("✅ SAFE CUSTOMER")

with col2:
    st.subheader("📊 Why do customers leave?")
    st.image(shap_img_path, use_column_width=True)
    st.caption("Top predictors based on SHAP analysis of historical data.")
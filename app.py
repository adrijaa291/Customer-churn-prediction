import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
with open("customer_churn_rfc.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
feature_names = data["feature_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("🔍 Customer Churn Prediction")

# Sidebar input
st.sidebar.header("Enter Customer Details")

def user_input():
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ['Yes', 'No'])
    dependents = st.sidebar.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ['Yes', 'No'])
    internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.sidebar.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 150.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'InternetService': internet_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Preprocess input using saved encoders
def preprocess_input(df):
    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])
    return df

processed = preprocess_input(input_df)

# Ensure all expected features are present
for col in feature_names:
    if col not in processed.columns:
        processed[col] = 0
processed = processed[feature_names]

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]
    st.markdown(f"### 📊 Prediction: {'Churn' if prediction else 'No Churn'}")
    st.markdown(f"### 🔢 Probability of Churn: {proba:.2%}")

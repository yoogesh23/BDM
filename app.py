import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Use this to load your trained model

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Credit Card Approval Prediction", layout="centered")

st.title("Credit Card Approval Prediction App")

st.write("""
This app predicts whether a customer will be **approved for a credit card** based on their details.
Please enter the following details:
""")

# Input fields
age = st.slider("Age", min_value=23, max_value=67, value=35)
experience = st.number_input("Experience (Years)", min_value=-3, max_value=43, value=10)
income = st.number_input("Annual Income (in thousands $)", min_value=8, max_value=224, value=60)
zip_code = st.text_input("ZIP Code", value="94000")

family = st.selectbox("Family Size", [1, 2, 3, 4], index=1)
ccavg = st.number_input("Credit Card Avg Monthly Spend ($)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
education = st.selectbox("Education Level", [1, 2, 3], format_func=lambda x: f"Education Level {x}")
mortgage = st.number_input("Mortgage ($)", min_value=0, max_value=635, value=0)

personal_loan = st.radio("Personal Loan?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
securities_account = st.radio("Securities Account?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
cd_account = st.radio("CD Account?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
online = st.radio("Online Banking?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

# Prepare input data
# Prepare input data
input_data = pd.DataFrame({
    "ID": [0],  # Dummy value to match the trained model's expected features
    "Age": [age],
    "Experience": [experience],
    "Income": [income],
    "ZIP Code": [int(zip_code) if zip_code.isdigit() else 0],
    "Family": [family],
    "CCAvg": [ccavg],
    "Education": [education],
    "Mortgage": [mortgage],
    "Personal Loan": [personal_loan],
    "Securities Account": [securities_account],
    "CD Account": [cd_account],
    "Online": [online]
})


# Prediction function
def predict_credit_card(input_df):
    # Use your actual model for prediction
    prediction = model.predict(input_df)[0]
    return prediction

# Prediction
if st.button("Predict"):
    prediction = predict_credit_card(input_data)
    if prediction == 1:
        st.success("The customer is likely to be **APPROVED** for a credit card!")
    else:
        st.error("The customer is **NOT APPROVED** for a credit card.")
    
    st.write("#### Customer Details")
    st.dataframe(input_data)

st.write("---")


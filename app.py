import streamlit as st
import pandas as pd
import joblib


# Page Configuration


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# Load Model


model = joblib.load("loan_decision_tree.pkl")


# Title

st.title("🏦 Loan Approval Prediction System")

st.write(
    """
This application predicts whether a loan is likely to be approved
based on applicant information using a trained Decision Tree model.
"""
)


# Sidebar


st.sidebar.header("Applicant Details")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.sidebar.number_input(
    "Annual Income",
    min_value=0,
    value=50000,
    step=1000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=0,
    value=20000,
    step=1000
)

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)


# Display Entered Data


st.subheader("Applicant Information")

data = pd.DataFrame({
    "Feature":[
        "Age",
        "Income",
        "Loan Amount",
        "Credit Score"
    ],
    "Value":[
        age,
        income,
        loan_amount,
        credit_score
    ]
})

st.table(data)

# Prediction
if st.button("Predict Loan Approval"):

    features = [[
        age,
        income,
        loan_amount,
        credit_score
    ]]

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)

    confidence = max(probability[0]) * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("✅ Loan Approved")

    else:

        st.error("❌ Loan Rejected")

    st.write(f"Confidence : {confidence:.2f}%")

    st.progress(int(confidence))

    st.divider()

    st.subheader("Entered Features")

    st.write(f"**Age:** {age}")

    st.write(f"**Income:** ₹ {income:,}")

    st.write(f"**Loan Amount:** ₹ {loan_amount:,}")

    st.write(f"**Credit Score:** {credit_score}")



st.markdown("---")

st.caption("Developed using Streamlit | Decision Tree Classifier")
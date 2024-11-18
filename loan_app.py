import pickle
import streamlit as st
import numpy as np

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@st.cache_data()
def prediction(Gender, Married, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History):   
    # Pre-processing user input
    Gender = 0 if Gender == "Male" else 1
    Married = 0 if Married == "Unmarried" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1
    LoanAmount = LoanAmount / 1000  # Convert to thousands as per original model expectations

    # Scaling the inputs
    scaled_inputs = scaler.transform([[Gender, Married, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History]])

    # Making predictions
    prediction = classifier.predict(scaled_inputs)
    pred = 'Approved' if prediction == 1 else 'Rejected'
    return pred

# Main function for the Streamlit app
def main():
    # Page layout
    st.markdown("<h1 style='text-align: center;'>Streamlit Loan Prediction ML App</h1>", unsafe_allow_html=True)

    # Space for an image
    st.image('loan_image.jpg', use_column_width=True)

    # Project overview section
    st.markdown("## Project Overview")
    st.markdown("""
    This project demonstrates a loan prediction model using a machine learning classifier.
    The model uses several independent variables like gender, marital status, income, and credit history 
    to predict whether a loan application will be approved or rejected.
    """)
    st.markdown("""
    Adjust the inputs below to see how the model predicts based on the provided data. 
    Inputs are pre-processed and scaled for consistent performance.
    """)

    # User input fields
    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married"))
    Credit_History = st.selectbox('Credit History', ("Unclear Debts", "No Unclear Debts"))
    ApplicantIncome = st.slider("Applicant's Monthly Income", 0, 100000, step=500)
    CoapplicantIncome = st.slider("Coapplicant's Monthly Income", 0, 50000, step=500)
    LoanAmount = st.slider("Loan Amount", 0, 500, step=5)

    # Predict button
    if st.button("Predict"):
        result = prediction(Gender, Married, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History)
        if result == "Approved":
            st.success("Congrats! Your loan is approved!!")
        else:
            st.error("Oh no! Looks like you won't be getting a loan.")

if __name__ == '__main__':
    main()



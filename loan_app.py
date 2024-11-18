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
    st.markdown("<h1 style='text-align: center;'>Loan Prediction App</h1>", unsafe_allow_html=True)

    # Space for an image
    st.image('loan_image.jpg', use_column_width=True)

    # Short project description
    st.write("""
    Analyzing loan datasets and building prediction models is essential for banks 
    to optimize loan approval decisions. By identifying patterns in historical data, 
    banks can make data-driven decisions, minimizing risks of defaults and improving 
    fairness and consistency in the loan approval process. This not only enhances 
    operational efficiency but also helps allocate resources effectively while reducing 
    financial losses.

    The aim of a loan prediction model is to assist banks in evaluating applicants' 
    repayment likelihood based on key factors like income and credit history. This 
    streamlines the approval process, saves time, and fosters financial inclusion by 
    identifying eligible low-risk applicants. The model ultimately creates a more 
    efficient and transparent lending system, benefiting both banks and customers.
    """)

    # User input fields
    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married"))
    Credit_History = st.selectbox('Credit History', ("Unclear Debts", "No Unclear Debts"))
    ApplicantIncome = st.slider("Applicant's Monthly Income", 0, 100000000, step=1)
    CoapplicantIncome = st.slider("Coapplicant's Monthly Income", 0, 50000000, step=1)
    LoanAmount = st.slider("Loan Amount", 0, 1000000000, step=1)

    # Predict button
    if st.button("Predict"):
        result = prediction(Gender, Married, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History)
        if result == "Approved":
            st.success("Congrats! Your loan is approved!!")
        else:
            st.error("Oh no! Looks like you won't be getting a loan.")

if __name__ == '__main__':
    main()



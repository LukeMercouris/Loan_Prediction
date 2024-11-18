import streamlit as st
import pickle
import numpy as np

def main():

    # Load the saved model
    with open('model.pkl', 'rb') as model_file:
        classifier = pickle.load(model_file)

    # Load the scaler
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Define the app title
    st.markdown("<h1 style='text-align: center;'>Loan Approval Prediction App</h1>", unsafe_allow_html=True)

    # Provide a description
    st.write("""
    This app uses a machine learning model to predict whether your loan will be approved or rejected.
    Please provide the necessary details below for the prediction.
    """)

    # Input fields for user data
    st.header("Enter the following details:")

    # Gender
    Gender = st.selectbox("Gender", ("Male", "Female"))

    # Marital Status
    Married = st.selectbox("Marital Status", ("Unmarried", "Married"))

    # Applicant Income
    ApplicantIncome = st.slider("Applicant's Monthly Income ($)", min_value=0, max_value=100000, step=1000)

    # Coapplicant Income
    CoapplicantIncome = st.slider("Coapplicant's Monthly Income ($)", min_value=0, max_value=50000, step=500)

    # Loan Amount
    LoanAmount = st.slider("Loan Amount (in thousands)", min_value=0, max_value=1000, step=1)

    # Credit History
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "No Unclear Debts"))

    # Display the final input values
    st.write("### Selected Values:")
    st.write(f"Gender: {Gender}")
    st.write(f"Marital Status: {Married}")
    st.write(f"Applicant's Monthly Income: {ApplicantIncome}")
    st.write(f"Coapplicant's Monthly Income: {CoapplicantIncome}")
    st.write(f"Loan Amount: {LoanAmount}")
    st.write(f"Credit History: {Credit_History}")

    # Button for final prediction
    if st.button("Predict"):
        # Preprocess the categorical inputs
        if Gender == "Male":
            Gender = 0
        else:
            Gender = 1

        if Married == "Unmarried":
            Married = 0
        else:
            Married = 1

        if Credit_History == "Unclear Debts":
            Credit_History = 0
        else:
            Credit_History = 1

        # Prepare the input data as an array
        input_data = np.array([[Gender, Married, ApplicantIncome, CoapplicantIncome, LoanAmount * 1000, Credit_History]])

        # Scale the input data using the scaler
        try:
            scaled_input_data = scaler.transform(input_data)
        except ValueError as e:
            st.error(f"Error with scaling input: {e}")
            return
        
        # Debug: Display scaled input data
        st.write("### Scaled Input Values (for Debugging):")
        st.write(scaled_input_data)

        # Make a prediction using the loaded model
        try:
            prediction = classifier.predict(scaled_input_data)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return
        
        # Display the prediction result
        if prediction[0] == 1:
            st.write("**CONGRATULATIONS!!!** Your loan is approved!")
        else:
            st.write("**SORRY!!!** Your loan is rejected.")

if __name__ == '__main__':
    main()



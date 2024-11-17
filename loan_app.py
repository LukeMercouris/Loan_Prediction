import pickle
import streamlit as st
import numpy as np

# Loading the trained model
pickle_in = open('model.pkl', 'rb') 
classifier = pickle.load(pickle_in)

# Loading the scaler
scaler_in = open('scaler.pkl', 'rb') 
scaler = pickle.load(scaler_in)
  
@st.cache_data()
# Defining the function to make predictions using user inputs
def prediction(Gender, Married, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History):   
 
    # Preprocessing user input    
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
 
    # Scaling the numerical variables
    # Ensure input is reshaped properly for the scaler
    scaled_values = scaler.transform(np.array([[ApplicantIncome, CoapplicantIncome, LoanAmount]]))
    ApplicantIncome, CoapplicantIncome, LoanAmount = scaled_values[0]
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History]])
     
    if prediction[0] == 0:  # Fix for prediction output
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred
  
  
# Main function for the webpage  
def main():       
    # Front end elements of the web page (removed background for title)
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Loan Prediction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    # Input fields for user data
    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married")) 

    # Applicant Income (slider)
    ApplicantIncome = st.slider("Applicant's Monthly Income ($)", min_value=0, max_value=100000, step=1000)

    # Coapplicant Income (slider)
    CoapplicantIncome = st.slider("Coapplicant's Monthly Income ($)", min_value=0, max_value=50000, step=500)

    # Loan Amount (slider)
    LoanAmount = st.slider("Total Loan Amount (in thousands)", min_value=0, max_value=1000, step=1)

    Credit_History = st.selectbox('Credit History', ("Unclear Debts", "No Unclear Debts"))
    result = ""
      
    # When 'Predict' is clicked, make the prediction and display the result
    if st.button("Predict"): 
        result = prediction(Gender, Married, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History) 
        st.success('Your loan is {}'.format(result))
     
if __name__ == '__main__': 
    main()



import pickle
import streamlit as st
import numpy as np

# Loading the trained model
with open('classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

# Loading the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@st.cache_data()
# Defining the function to make predictions using user inputs
def prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History):   
 
    # Pre-processing user input    
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
    input_data = np.array([[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
    
    # Scale the input data using the scaler
    scaled_input = scaler.transform(input_data)
 
    # Making predictions 
    prediction = classifier.predict(scaled_input)
     
    if prediction[0] == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred
      
  
# Main function for the webpage  
def main():       
    # Front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # Display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True) 
      
    # Input fields for user data
    Gender = st.selectbox('Gender', ("Male", "Female"))
    Married = st.selectbox('Marital Status', ("Unmarried", "Married")) 
    ApplicantIncome = st.number_input("Applicant's Monthly Income ($)") 
    LoanAmount = st.number_input("Total Loan Amount ($)")
    Credit_History = st.selectbox('Credit History', ("Unclear Debts", "No Unclear Debts"))
    result = ""
      
    # When 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History) 
        st.success('Your loan is {}'.format(result))
     
if __name__ == '__main__': 
    main()



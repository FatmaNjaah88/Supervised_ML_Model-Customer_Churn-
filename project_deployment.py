import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

#upload data

model=pickle.load(open(r"E:\faculty\Senior Year\AI &Robertics(IT403)\AI's Project\Python code\Deployment Model\Customer_Chun_predict.sav",'rb'))
scaler = pickle.load(open(r"E:\faculty\Senior Year\AI &Robertics(IT403)\AI's Project\Python code\Deployment Model\scaler.sav", 'rb'))


st.title("Customer Churn Prediction App")
st.info("AI model trained to predict if the customer will exit or not")

columns =['CreditScore', 'Age', 'Tenure','Balance', 'HasCrCard', 'IsActiveMember',
           'EstimatedSalary', 'Geography_France', 'Geography_Germany',
           'Geography_Spain', 'Gender_Female', 'Gender_Male',
            
           'NumOfProducts_More than two products', 'NumOfProducts_One product',
           'NumOfProducts_Two products']


Age = st.number_input("Age", min_value=18, max_value=80)
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, step=0.01, format="%.2f")
CreditScore = st.number_input("Credit Score", min_value=300, max_value=900)
Gender = st.selectbox("Gender", ["Male", "Female"])
Balance = st.number_input("Balance", min_value=0.0)
Tenure = st.number_input("Tenure (Number of years with bank)", min_value=0, max_value=10)
HasCrCard = st.selectbox("Has Credit Card?", ["No", "Yes"])
IsActiveMember = st.selectbox("Is Active Member?", ["No", "Yes"])
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
NumOfProducts = st.selectbox("Number of Products", ["One product", "Two products", "More than two products"])

# Encode categorical inputs
user_values = {
    'CreditScore': CreditScore,
    'Age': Age,
    'Balance': Balance,
    'EstimatedSalary': EstimatedSalary,
    'Gender_Female': 1 if Gender == "Female" else 0,
    'Gender_Male': 1 if Gender == "Male" else 0,
    'Tenure': Tenure,
    'HasCrCard': 1 if HasCrCard == "Yes" else 0,
    'IsActiveMember': 1 if IsActiveMember == "Yes" else 0,
    'Geography_France': 1 if Geography == "France" else 0,
    'Geography_Germany': 1 if Geography == "Germany" else 0,
    'Geography_Spain': 1 if Geography == "Spain" else 0,
    'NumOfProducts_One product': 1 if NumOfProducts == "One product" else 0,
    'NumOfProducts_Two products': 1 if NumOfProducts == "Two products" else 0,
    'NumOfProducts_More than two products': 1 if NumOfProducts == "More than two products" else 0
}

input_list = [user_values[col] for col in columns]

# عمل DataFrame جاهز للـ scaler
input_data = pd.DataFrame([input_list], columns=columns)


if st.button("Predict"):
    # input_data = np.array([[Age, EstimatedSalary, Gender, Balance, CreditScore]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)[0]


    if result == 1:
        st.error("❌ The model predicts: **Customer Will Exit**")
        prob = model.predict_proba(input_scaled)[0][1]  # احتمال Exited=1
        st.write(f"Probability of Exit: {prob*100:.2f}%")

    else:
        st.success("✅ The model predicts: **Customer Will Stay**")
        prob = model.predict_proba(input_scaled)[0][1]  # احتمال Exited=1
        st.write(f"Probability of Exit: {prob*100:.2f}%")


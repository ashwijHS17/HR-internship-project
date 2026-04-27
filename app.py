import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# 1. Load the model
model = pickle.load(open('kr_model.pkl', 'rb')) 

# 2. App Title
st.title('HR Attrition Prediction App')
st.write("Enter employee details to predict the likelihood of attrition.")

# 3. Define Inputs based on Task_8-1 Features
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=60, value=30)
    monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000)
    distance = st.number_input('Distance From Home', min_value=1, max_value=30, value=5)
    total_working_years = st.number_input('Total Working Years', min_value=0, max_value=40, value=10)
    
with col2:
    job_level = st.slider('Job Level', 1, 5, 2)
    job_satisfaction = st.slider('Job Satisfaction (1-4)', 1, 4, 3)
    env_satisfaction = st.slider('Environment Satisfaction (1-4)', 1, 4, 3)
    overtime = st.selectbox('Overtime', ('Yes', 'No'))

# 4. Preprocessing & Encoding
overtime_val = 1 if overtime == 'Yes' else 0 

input_data = {
    'Age': age,
    'BusinessTravel': 1, 
    'DailyRate': 800,
    'Department': 1,
    'DistanceFromHome': distance,
    'Education': 3,
    'EducationField': 1,
    'EmployeeCount': 1,
    'EmployeeNumber': 1,
    'EnvironmentSatisfaction': env_satisfaction,
    'Gender': 1,
    'HourlyRate': 65,
    'JobInvolvement': 3,
    'JobLevel': job_level,
    'JobRole': 1,
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': 1,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': 14000,
    'NumCompaniesWorked': 1,
    'Over18': 1,
    'OverTime': overtime_val,
    'PercentSalaryHike': 15,
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 3,
    'StandardHours': 80, 
    'StockOptionLevel': 0,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': 2,
    'WorkLifeBalance': 3,
    'YearsAtCompany': 5,
    'YearsInCurrentRole': 2,
    'YearsSinceLastPromotion': 1,
    'YearsWithCurrManager': 2
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

column_order = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 
                'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 
                'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 
                'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 
                'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 
                'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 
                'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
                'YearsSinceLastPromotion', 'YearsWithCurrManager']

input_df = input_df[column_order]

# Prediction
if st.button('Predict Attrition'): 
    prediction = model.predict(input_df)
   
    if prediction[0] == 1:
        st.error('The model predicts this employee is LIKELY to leave.')
    else:
        st.success('The model predicts this employee is LIKELY to stay.')
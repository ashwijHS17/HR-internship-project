import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Page config (NEW)
st.set_page_config(page_title="HR Attrition Predictor", page_icon="👨‍💼", layout="wide")

# Load model
model = pickle.load(open('kr_model.pkl', 'rb'))

# Custom CSS (NEW UI)
st.markdown("""
    <style>
    .main-title {text-align:center; font-size:40px; font-weight:bold; color:#2E86C1;}
    .sub-title {text-align:center; font-size:18px; color:gray;}
    .result-box {padding:20px; border-radius:15px; text-align:center; font-size:22px;}
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown("<p class='main-title'>HR Attrition Prediction App</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict whether an employee is likely to leave the company</p>", unsafe_allow_html=True)

st.write("")

# Sidebar inputs (UPDATED)
st.sidebar.header("Enter Employee Details")

age = st.sidebar.number_input('Age', 18, 60, 30)
monthly_income = st.sidebar.number_input('Monthly Income', 1000, 20000, 5000)
distance = st.sidebar.number_input('Distance From Home', 1, 30, 5)
total_working_years = st.sidebar.number_input('Total Working Years', 0, 40, 10)

job_level = st.sidebar.slider('Job Level', 1, 5, 2)
job_satisfaction = st.sidebar.slider('Job Satisfaction', 1, 4, 3)
env_satisfaction = st.sidebar.slider('Environment Satisfaction', 1, 4, 3)
overtime = st.sidebar.selectbox('Overtime', ('Yes', 'No'))

st.write("### Click below to predict")

# Encoding
overtime_val = 1 if overtime == 'Yes' else 0

input_data = {
    'Age': age, 'BusinessTravel': 1, 'DailyRate': 800, 'Department': 1,
    'DistanceFromHome': distance, 'Education': 3, 'EducationField': 1,
    'EmployeeCount': 1, 'EmployeeNumber': 1,
    'EnvironmentSatisfaction': env_satisfaction, 'Gender': 1,
    'HourlyRate': 65, 'JobInvolvement': 3, 'JobLevel': job_level,
    'JobRole': 1, 'JobSatisfaction': job_satisfaction, 'MaritalStatus': 1,
    'MonthlyIncome': monthly_income, 'MonthlyRate': 14000,
    'NumCompaniesWorked': 1, 'Over18': 1, 'OverTime': overtime_val,
    'PercentSalaryHike': 15, 'PerformanceRating': 3,
    'RelationshipSatisfaction': 3, 'StandardHours': 80,
    'StockOptionLevel': 0, 'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': 2, 'WorkLifeBalance': 3,
    'YearsAtCompany': 5, 'YearsInCurrentRole': 2,
    'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 2
}

input_df = pd.DataFrame([input_data])

column_order = ['Age','BusinessTravel','DailyRate','Department','DistanceFromHome',
'Education','EducationField','EmployeeCount','EmployeeNumber','EnvironmentSatisfaction',
'Gender','HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction',
'MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime',
'PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours',
'StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance',
'YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']

input_df = input_df[column_order]

# Prediction button (CENTERED)
col1, col2, col3 = st.columns([1,2,1])
predict_btn = col2.button("🔍 Predict Attrition")

if predict_btn:
    prediction = model.predict(input_df)

    st.write("")
    if prediction[0] == 1:
        st.markdown(
            "<div class='result-box' style='background-color:#FADBD8; color:#C0392B;'>⚠️ Employee is LIKELY to leave</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box' style='background-color:#D5F5E3; color:#1E8449;'>✅ Employee is LIKELY to stay</div>",
            unsafe_allow_html=True
        )

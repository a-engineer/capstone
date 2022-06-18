import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white
from statsmodels.compat import lzip
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st
import streamlit_authenticator as stauth
import statsmodels.api as sm

header  = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

add_selectbox = st.sidebar.selectbox("How would you like to be contacted?",("Email", "Home phone", "Mobile phone"))

# Using "with" notation
# with st.sidebar:
add_radio = st.sidebar.radio("Choose a shipping method",("Standard (5-15 days)", "Express (2-5 days)"))

with header:
    st.markdown("<h1 style='text-align: center; color: black;'>Welcome to my Data Science Capstone!</h1>", unsafe_allow_html=True) 
    st.text('In this project, I will take historical employee productivity data and create a \nmultiple linear regression model to be able to predict employee productivity.')

with dataset: 
    st.header('Employee Productivity Data') 
    st.text('This data has nine columns with the following information:') 
    st.markdown('* employee = In the real data this would actually be the employee\'s name.') 
    st.markdown('* reg_hours = This represents the number of regular hours that an employee worked.') 
    st.markdown('* ot_hours = This represents the number of overtime that an employee worked.') 
    st.markdown('* vac_hours = This represents the number of vacation hours that an employee took.') 
    st.markdown('* task1_hrs = This task represents the amount of time an employee spent on new business work.') 
    st.markdown('* task2_hrs = This task represents the amount of time an employee spent on maintenance work.') 
    st.markdown('* task3_hrs = This task represents the amount of time an employee spent on training.') 
    st.markdown('* task4_hrs = This task represents the amount of time an employee spent on emergenecy work.') 
    st.markdown('* productivity = This represents the total yearly productivity of that employee.')

st.subheader('Choose a file to upload or Select Example Dataset Button.')

if st.button("Example Dataset"):
    df = pd.read_csv("employee productivity v2.csv") 

else:
    uploaded_file = st.file_uploader("Choose a file") 
    if uploaded_file is not None:       
        # Cache dataset
        @st.cache
        def get_data():
            # Reads an uploaded file
            df = pd.read_csv(uploaded_file)
            return df

        # Displays the first five rows
        df = get_data()
        st.subheader('First five rows of data.')
        st.table(df.head())





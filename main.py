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


add_selectbox = st.sidebar.selectbox("How would you like to be contacted?",("Email", "Home phone", "Mobile phone"))

# Using "with" notation
# with st.sidebar:
add_radio = st.sidebar.radio("Choose a shipping method",("Standard (5-15 days)", "Express (2-5 days)"))

with header:
    st.markdown("<h1 style='text-align: center; color: black;'>Welcome to my Data Science Capstone!</h1>", unsafe_allow_html=True) 
    st.text('In this project, I will take historical employee productivity data and create a multiple linear regression model to be able to predict employee productivity.')
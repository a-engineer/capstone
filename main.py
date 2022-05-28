import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

#---------Password Login------------
names = ['John Smith','Rebecca Briggs']
usernames = ['jsmith','rbriggs']
passwords = ['123','456']

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
    'some_cookie_name','some_signature_key',cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login','sidebar')

if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write('Welcome *%s*' % (name))
    header  = st.container()
    dataset = st.container()
    features = st.container()
    modelTraining = st.container()
    with header: st.markdown("<h1 style='text-align: center; color: black;'>Welcome to my Data Science Capstone!</h1>", unsafe_allow_html=True)
    with header: st.text('''In this project, I will take historical employee productivity data and create a 
multiple linear regression model to be able to predict employee productivity.''')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')


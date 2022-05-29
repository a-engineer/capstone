from lib2to3.pgen2.token import NEWLINE
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

header  = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

if authentication_status == False:
	st.error('Username/password is incorrect')

if authentication_status == None:
	st.warning('Please enter your username and password')

if authentication_status:
    st.sidebar.write('Welcome *%s*' % (name))
    authenticator.logout('Logout', 'sidebar')
    # Using object notation
    add_selectbox = st.sidebar.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone")
         )

    # Using "with" notation
    #with st.sidebar:
    add_radio = st.sidebar.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
        )
    with header: 
        st.markdown("<h1 style='text-align: center; color: black;'>Welcome to my Data Science Capstone!</h1>", unsafe_allow_html=True) 
        st.text('''In this project, I will take historical employee productivity data
and create a multiple linear regression model to be able to
predict employee productivity.''')
    with dataset: 
        st.header('Employee Productivity Data') 
        st.text('''This data has nine columns with the following information:''') 
        st.markdown('* employee = In the real data this would actually be the employee\'s name.''') 
        st.markdown('* reg_hours = This represents the number of regular hours that an employee worked.') 
        st.markdown('* ot_hours = This represents the number of overtime that an employee worked.') 
        st.markdown('* vac_hours = This represents the number of vacation hours that an employee took.') 
        st.markdown('* task1_hrs = This task represents the amount of time an employee spent on new business work.') 
        st.markdown('* task2_hrs = This task represents the amount of time an employee spent on maintenance work.') 
        st.markdown('* task3_hrs = This task represents the amount of time an employee spent on training.') 
        st.markdown('* task4_hrs = This task represents the amount of time an employee spent on emergenecy work.') 
        st.markdown('* productivity = This represents the total yearly productivity of that employee.')
    
    st.subheader('Choose a file to upload.') 
    uploaded_file = st.file_uploader("Choose a file") 
    if uploaded_file is not None:
        
        # Reads an uploaded file
        df = pd.read_csv(uploaded_file)

        # Displays the first five rows
        st.subheader('Uploaded files first five rows of data.')
        st.table(df.head())
        
        # Displays the null values
        st.subheader("Number of null values.")
        nullvalues = df.isnull().sum()
        st.write(nullvalues)

        # Counts number of columns/variables
        x = df.columns
        x_len = len(x)

        # Uses length of of columns to create a checkbox for loop
        import array as arr
        st.subheader("Select Independent Variables:")
        z = []
        y = []
        for y in range(x_len):
            z.append(st.checkbox(x[y]))
        z[0]
        z[1]
            


        
            



    

    
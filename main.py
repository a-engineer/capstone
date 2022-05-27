import streamlit as st
import pandas as pd


header  = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header:
	st.markdown("<h1 style='text-align: center; color: black;'>Welcome to my Data Science Capstone!</h1>", unsafe_allow_html=True)

	#st.title('Welcome to my Data Science Capstone!')
	st.text('''In this project, I will take historical employee productivity data
and create a multiple linear regression model to be able to
predict employee productivity.
	''')

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

	uploaded_file = st.file_uploader("Choose a file")
	if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
    		df = pd.read_csv(uploaded_file)
    		st.write(df.head())

	st.subheader('Uploaded file\'s first five rows of data.')

	taxi_data = pd.read_csv('Tutorial/data/employee productivity v2.csv')
	
	st.subheader('The following is the first five rows of the dataset.')
	st.write(taxi_data.head())

	st.subheader('Number of null values.')
	nullvalues = taxi_data.isnull().sum()
	st.write(nullvalues)

with features:
	st.header('The features I created for this project.')

with modelTraining:
	st.header('Time to train the model!')
	st.text('Description of variables.')

sel_col, disp_col = st.columns(2)

max_depth = sel_col.slider('Select how many variables you would like regression to include:',min_value = 1, max_value = 10,value = 1, step=1)

n_estimators = sel_col.selectbox('How many variables should there be?',options=[100,200,300,'No Limit'],index = 0)

input_feature = sel_col.text_input("Which feature should be used as the input feature?")
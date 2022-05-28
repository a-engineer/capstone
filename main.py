import streamlit as st

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
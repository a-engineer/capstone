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


# Using object notation
#add_selectbox = st.sidebar.selectbox("Select Machine Learning Algorithm",("Linear Regression", "Multiple Linear Regression", "Logistic Regression"))

# Using "with" notation
#with st.sidebar:
add_radio = st.sidebar.radio("Select Machine Learning Algorithm (Future Update)",("Linear Regression", "Multiple Linear Regression","Logistic Regression"))
    
with header: 
    st.markdown("<h1 style='text-align: center; color: black;'>Welcome to my Data Science Capstone!</h1>", unsafe_allow_html=True) 
    st.text('''In this project, I will take historical employee productivity data
and create a multiple linear regression model to be able to
predict employee productivity.''')
    
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
    
    st.subheader("Select Sample Data or Upload Data")
    
    if st.button('Sample Data'):
        df = pd.read_csv("employee productivity v2.csv")
        st.subheader('Uploaded files first five rows of data.')
        st.table(df.head())

        # Counts number of columns/variables
        col_nam = df.columns
        col_len = len(col_nam)

        # See the data types for each variable
        st.subheader('Data Types')
        d = []
        cat_var = []
        for c in range(col_len):
            st.write(col_nam[c])
            st.write(df[col_nam[c]].dtype)       
            d.append(df[col_nam[c]].dtype)
            if d[c] == 'object':
                cat_var.append(col_nam[c])

        # Displays the null values
        st.subheader("Number of null values.")
        nullvalues = df.isnull().sum()
        st.write(nullvalues)

        # How to handle missing values
        # https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
        st.subheader("Select Option To Deal With Null Values")
        option = st.selectbox(
            "Select Null Value Imputation Method",
            ('Delete Row','Replace W/ Mean/Median','Deep Learning-Datawig'))
        st.write('You Selected:', option)

        # Displays the Distribution of Numerical Features
        st.subheader("Distribution of Numerical Features")
        st.write(df.describe())

        # Displays the Distribution of Numerical Features
        #st.subheader("Distribution of Categorical Features")
        #st.write(df.describe(include=['O']))

        # Displays the Boxplot of Outliers
        st.subheader("Outlier Detection Using Box Plot")
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=df, orient="h", palette="Set2")
        st.pyplot(fig)

        # Categorical Variables
        st.subheader("Categorical Variables")
        if not cat_var: #Checks to see if cat_var is empty
            st.write("No Categorical Variables") # If empty then it tells user
        else: # If cat_var is not empty then write the variables
            st.table(cat_var)

        # Uses length of of columns to create a checkbox for loop
        import array as arr
        st.subheader("Select Independent Variables:")
        z = []
        y = []
        ind_v = []
        for y in range(col_len):
            z.append(st.checkbox(col_nam[y]))
            if z[y] == True:
                ind_v.append(col_nam[y])
        st.subheader("Select Dependent Variables:")
        a = []
        b = []
        dep_v = []
        count = 0
        for b in range(col_len):
            a.append(st.checkbox(col_nam[b],key = b))
            count += 1
            if a[b] == True:
                dep_v.append(col_nam[b])
        st.subheader("Independent Variables:")
        st.table(ind_v)
        st.subheader("Dependent Variables:")
        st.table(dep_v)

        X = df[ind_v]
        #X = sm.add_constant(X)
        y = df[dep_v]

        st.header("Scaling Data")
        st.markdown("Before scaling the indepedent variables:")
        st.write(X)
        sc = StandardScaler()
        X_sc = pd.DataFrame(sc.fit_transform(X),columns = X.columns)
        X_sc = sm.add_constant(X_sc)
        st.markdown("After scaling the indepedent variables and adding a constant:")
        st.write(X_sc)
        # The following creates a new dataframe with the selected
        # independent and dependent variables. It creates a df3
        # dataframe and joins the ind_v and dep_v dataframes.
        #df3 = X_sc.tolist()
        df3 = X_sc.join(y)

        # Splitting the data into training and testing.
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_sc,y, test_size = 0.2, random_state = 42)

        # Applying Linear Regression Model v2 from sklearn
        #from sklearn import linear_model
        #LR = linear_model.LinearRegression()
        #model2 = LR.fit(X_train,y_train)
        # Applying Linear Regression Model from sklearn
        from sklearn.linear_model import LinearRegression
        LR = LinearRegression()
        model2 = LR.fit(X_train, y_train)
        model = sm.OLS(y_train, X_train).fit()

        # Compares actual values with predicted values
        
        y_prediction = LR.predict(X_test)
        predictions = model.predict(X_sc)
        resids = model.resid
        stand_resid = model.get_influence().resid_studentized_internal

        y_pred = LR.predict(X_train)
        residuals = y_train.values-y_pred
        fig, ax = plt.subplots()
        #ax = sns.scatterplot(x = y_pred,y = residuals)

        # Multiple Linear Regression Assumption Testing
        st.header("Assumption Testing")
        st.text("The following will test the five assumptions of multiple linear regression.")
        st.markdown("1. Linear Relationship \n 2. No Multicollinearity \n 3. Independence \n 4. Homoscedasticity \
        \n 5. Multivariate Normality ")

        st.subheader("1. Linear Relationship Assumption")
        st.text("The following plots will graph each indepedent variable vs the dependent variable.")
        st.text("You will want to ensure there is a linear relationship present in each graph.")
        st.pyplot(sns.pairplot(df3, x_vars=ind_v,y_vars = dep_v))
        for i in range(len(ind_v)):
            g = sns.JointGrid(data=df3, x=ind_v[i],y=dep_v[0])
            st.pyplot(g.plot(sns.regplot,sns.histplot))

        st.subheader("2a. Multicollinearity Assumption")
        st.text("The following tests multicollinearity among variables by using a \ncorrelation heatmap.")
        st.text("You will want to ensure there is no multicollinearity greater than 0.8.")
        
        # Displays the Correlation Heatmap
        st.text("Correlation Heatmap")
        corr_map = df3.drop(columns=['const'])
        fig, ax = plt.subplots(figsize=(10,10))
        ax = sns.heatmap(corr_map.corr(), annot=True,linewidths=0.5)
        st.pyplot(fig)

        # Displays the Variance Inflation Factor (VIF) values
        st.subheader("2b. Multicollinearity Assumption \nVariance Inflation Factor (VIF)")
        st.text("Another test for multicollinearity is to identify the Variance Inflation \nFactor (VIF) values.")
        st.text("A VIF value greater than 5 indicates multicollinearity is present. \nVariables with \
a value greater than 5 should be deleted or a ridge or lasso \nregression should be used instead.")
        st.text("Ignore the const variable since that is just the intercept.")
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X_sc.values, i) for i in range(X_sc.shape[1])]
        vif['variable'] = X_sc.columns
        st.write(vif)

        # Independence Assumption Testing
        st.subheader("3. Independence Assumption Testing")
        st.text("To test for Independence, the Durbin-Watson test will be used.")
        st.text("The Durbin-Watson test uses the following hypotheses:")
        st.markdown("1. H_0 (null hypothesis): No correlation among the residuals and\n 2. H_A \
            (alternative hypothesis): The residuals are autocorrelated.")
        st.text("An acceptable and passing Durbin-Watson value is from 1.5 to 2.5.")
        durb = durbin_watson(model.resid)
        if 1.5 < durb < 2.5:
            durb_text = '<p style="color:red;">Passed!</p>'
            st.write("Durbin-Watson Score: ", durb, )
            st.markdown(durb_text, unsafe_allow_html=True)

        # Homoscedasticity Assumption Testing
        st.subheader("4. Homoscedasticity Assumption Testing")
        
        st.markdown("Residuals vs Predictions Plot")
        fig, ax = plt.subplots()
        plt.scatter(y_pred, stand_resid)
        plt.xlabel('Predictions')
        plt.ylabel('Standardized Residuals')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
        st.pyplot(fig)

        name = ['Lagrange multiplier statistic', 'p-value','f-value','f p-value']
        test = sms.het_breuschpagan(model.resid, model.model.exog)
        results = lzip(name, test)
        st.markdown("Breusch-Pagan Test")
        st.table(results)

        # White's Test
        st.markdown("White's Test")
        white_test = het_white(model.resid,  model.model.exog)
        labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
        results2 = lzip(labels,white_test)
        st.table(results2)

        # Multivariate Normality Assumption Testing
        st.subheader("5a. Multivariate Normality Assumption Testing")
        st.subheader("Histogram")
        fig = sns.displot(resids,kde = True)
        plt.title("Normality of Residuals")
        st.pyplot(fig)

        st.subheader("5b. Multivariate Normality Assumption Testing")
        st.subheader("Q-Q Plot")
        st.text("To test for normality, a Q-Q (Quantile-Quantile) plot is used to see if \nthe points lie on a straight diagonal line.")
        fig = sm.qqplot(resids, line='45',fit=True,dist=stats.norm)
        st.pyplot(fig)
        
        # Model Summary Results
        st.subheader("Model Evaluation Summary Results")
        st.write("MODEL 1")
        st.write(model.summary())

        # Performance Metrics
        from sklearn import metrics
        from sklearn.metrics import r2_score
        meanAbErr = metrics.mean_absolute_error(y_test, y_prediction)
        meanSqErr = metrics.mean_squared_error(y_test, y_prediction)
        rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_prediction))
        st.subheader("Performance Metrics")
        st.write('R squared: ', r2_score(y_test, y_prediction))
        st.write('Mean Absolute Error:', meanAbErr)
        st.write('Mean Square Error:', meanSqErr)
        st.write('Root Mean Square Error:', rootMeanSqErr)
    else:
        st.subheader('Choose a file to upload.') 
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
            st.subheader('Uploaded files first five rows of data.')
            st.table(df.head())

            # Counts number of columns/variables
            col_nam = df.columns
            col_len = len(col_nam)

            # See the data types for each variable
            st.subheader('Data Types')
            d = []
            cat_var = []
            for c in range(col_len):
                st.write(col_nam[c])
                st.write(df[col_nam[c]].dtype)       
                d.append(df[col_nam[c]].dtype)
                if d[c] == 'object':
                    cat_var.append(col_nam[c])

            # Displays the null values
            st.subheader("Number of null values.")
            nullvalues = df.isnull().sum()
            st.write(nullvalues)

            # How to handle missing values
            # https://towardsdatascience.com/7-ways-to-handle-missing-values-in-machine-learning-1a6326adf79e
            st.subheader("Select Option To Deal With Null Values")
            option = st.selectbox(
                "Select Null Value Imputation Method",
                ('Delete Row','Replace W/ Mean/Median','Deep Learning-Datawig'))
            st.write('You Selected:', option)

            # Displays the Distribution of Numerical Features
            st.subheader("Distribution of Numerical Features")
            st.write(df.describe())

            # Displays the Distribution of Numerical Features
            #st.subheader("Distribution of Categorical Features")
            #st.write(df.describe(include=['O']))

            # Displays the Boxplot of Outliers
            st.subheader("Outlier Detection Using Box Plot")
            fig, ax = plt.subplots()
            ax = sns.boxplot(data=df, orient="h", palette="Set2")
            st.pyplot(fig)

            # Categorical Variables
            st.subheader("Categorical Variables")
            if not cat_var: #Checks to see if cat_var is empty
                st.write("No Categorical Variables") # If empty then it tells user
            else: # If cat_var is not empty then write the variables
                st.table(cat_var)

            # Uses length of of columns to create a checkbox for loop
            import array as arr
            st.subheader("Select Independent Variables:")
            z = []
            y = []
            ind_v = []
            for y in range(col_len):
                z.append(st.checkbox(col_nam[y]))
                if z[y] == True:
                    ind_v.append(col_nam[y])
            st.subheader("Select Dependent Variables:")
            a = []
            b = []
            dep_v = []
            count = 0
            for b in range(col_len):
                a.append(st.checkbox(col_nam[b],key = b))
                count += 1
                if a[b] == True:
                    dep_v.append(col_nam[b])
            st.subheader("Independent Variables:")
            st.table(ind_v)
            st.subheader("Dependent Variables:")
            st.table(dep_v)

            X = df[ind_v]
            #X = sm.add_constant(X)
            y = df[dep_v]

            st.header("Scaling Data")
            st.markdown("Before scaling the indepedent variables:")
            st.write(X)
            sc = StandardScaler()
            X_sc = pd.DataFrame(sc.fit_transform(X),columns = X.columns)
            X_sc = sm.add_constant(X_sc)
            st.markdown("After scaling the indepedent variables and adding a constant:")
            st.write(X_sc)
            # The following creates a new dataframe with the selected
            # independent and dependent variables. It creates a df3
            # dataframe and joins the ind_v and dep_v dataframes.
            #df3 = X_sc.tolist()
            df3 = X_sc.join(y)

            # Splitting the data into training and testing.
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_sc,y, test_size = 0.2, random_state = 42)

            # Applying Linear Regression Model v2 from sklearn
            #from sklearn import linear_model
            #LR = linear_model.LinearRegression()
            #model2 = LR.fit(X_train,y_train)
            # Applying Linear Regression Model from sklearn
            from sklearn.linear_model import LinearRegression
            LR = LinearRegression()
            model2 = LR.fit(X_train, y_train)
            model = sm.OLS(y_train, X_train).fit()

            # Compares actual values with predicted values
            
            y_prediction = LR.predict(X_test)
            predictions = model.predict(X_sc)
            resids = model.resid
            stand_resid = model.get_influence().resid_studentized_internal

            y_pred = LR.predict(X_train)
            residuals = y_train.values-y_pred
            fig, ax = plt.subplots()
            #ax = sns.scatterplot(x = y_pred,y = residuals)

            # Multiple Linear Regression Assumption Testing
            st.header("Assumption Testing")
            st.text("The following will test the five assumptions of multiple linear regression.")
            st.markdown("1. Linear Relationship \n 2. No Multicollinearity \n 3. Independence \n 4. Homoscedasticity \
            \n 5. Multivariate Normality ")

            st.subheader("1. Linear Relationship Assumption")
            st.text("The following plots will graph each indepedent variable vs the dependent variable.")
            st.text("You will want to ensure there is a linear relationship present in each graph.")
            st.pyplot(sns.pairplot(df3, x_vars=ind_v,y_vars = dep_v))
            for i in range(len(ind_v)):
                g = sns.JointGrid(data=df3, x=ind_v[i],y=dep_v[0])
                st.pyplot(g.plot(sns.regplot,sns.histplot))

            st.subheader("2a. Multicollinearity Assumption")
            st.text("The following tests multicollinearity among variables by using a \ncorrelation heatmap.")
            st.text("You will want to ensure there is no multicollinearity greater than 0.8.")
            
            # Displays the Correlation Heatmap
            st.text("Correlation Heatmap")
            corr_map = df3.drop(columns=['const'])
            fig, ax = plt.subplots(figsize=(10,10))
            ax = sns.heatmap(corr_map.corr(), annot=True,linewidths=0.5)
            st.pyplot(fig)

            # Displays the Variance Inflation Factor (VIF) values
            st.subheader("2b. Multicollinearity Assumption \nVariance Inflation Factor (VIF)")
            st.text("Another test for multicollinearity is to identify the Variance Inflation \nFactor (VIF) values.")
            st.text("A VIF value greater than 5 indicates multicollinearity is present. \nVariables with \
    a value greater than 5 should be deleted or a ridge or lasso \nregression should be used instead.")
            st.text("Ignore the const variable since that is just the intercept.")
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X_sc.values, i) for i in range(X_sc.shape[1])]
            vif['variable'] = X_sc.columns
            st.write(vif)

            # Independence Assumption Testing
            st.subheader("3. Independence Assumption Testing")
            st.text("To test for Independence, the Durbin-Watson test will be used.")
            st.text("The Durbin-Watson test uses the following hypotheses:")
            st.markdown("1. H_0 (null hypothesis): No correlation among the residuals and\n 2. H_A \
                (alternative hypothesis): The residuals are autocorrelated.")
            st.text("An acceptable and passing Durbin-Watson value is from 1.5 to 2.5.")
            durb = durbin_watson(model.resid)
            if 1.5 < durb < 2.5:
                durb_text = '<p style="color:red;">Passed!</p>'
                st.write("Durbin-Watson Score: ", durb, )
                st.markdown(durb_text, unsafe_allow_html=True)

            # Homoscedasticity Assumption Testing
            st.subheader("4. Homoscedasticity Assumption Testing")
            
            st.markdown("Residuals vs Predictions Plot")
            fig, ax = plt.subplots()
            plt.scatter(y_pred, stand_resid)
            plt.xlabel('Predictions')
            plt.ylabel('Standardized Residuals')
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
            st.pyplot(fig)

            name = ['Lagrange multiplier statistic', 'p-value','f-value','f p-value']
            test = sms.het_breuschpagan(model.resid, model.model.exog)
            results = lzip(name, test)
            st.markdown("Breusch-Pagan Test")
            st.table(results)

            # White's Test
            st.markdown("White's Test")
            white_test = het_white(model.resid,  model.model.exog)
            labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
            results2 = lzip(labels,white_test)
            st.table(results2)

            # Multivariate Normality Assumption Testing
            st.subheader("5a. Multivariate Normality Assumption Testing")
            st.subheader("Histogram")
            fig = sns.displot(resids,kde = True)
            plt.title("Normality of Residuals")
            st.pyplot(fig)

            st.subheader("5b. Multivariate Normality Assumption Testing")
            st.subheader("Q-Q Plot")
            st.text("To test for normality, a Q-Q (Quantile-Quantile) plot is used to see if \nthe points lie on a straight diagonal line.")
            fig = sm.qqplot(resids, line='45',fit=True,dist=stats.norm)
            st.pyplot(fig)
            
            # Model Summary Results
            st.subheader("Model Evaluation Summary Results")
            st.write("MODEL 1")
            st.write(model.summary())

            # Performance Metrics
            from sklearn import metrics
            from sklearn.metrics import r2_score
            meanAbErr = metrics.mean_absolute_error(y_test, y_prediction)
            meanSqErr = metrics.mean_squared_error(y_test, y_prediction)
            rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_prediction))
            st.subheader("Performance Metrics")
            st.write('R squared: ', r2_score(y_test, y_prediction))
            st.write('Mean Absolute Error:', meanAbErr)
            st.write('Mean Square Error:', meanSqErr)
            st.write('Root Mean Square Error:', rootMeanSqErr)
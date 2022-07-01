# Data Science Capstone - Employee Productivity Projection

This project aims to predict employee producitivity by using historical data and by employing a multiple linear regression model.

**Business Problem**:

The author currently performs employee productivity projections.  These projections are used to budget additional labor and material.  However, the current method is to use a three year average to predict future projections.  The problem with this method is that the data has outliers and null values that will cause the projections to be over and under by a significant amount.  This results in not enough material being ordered or too much being ordered.  It also results in hiring too many employees or not enough employees to meet the projected productivity demand.

**Business Solution**:

As a result, the author decided to apply a multiple linear regression model to increase the accuracy of predicting employee productivity.  The author will clean the data, identify outliers, identify correlations, remove null values, perform assumption testing, and output performance metrics to help the user identify the accuracy of the model.  With this new method the author hopes to increase the accuracy of the employee productivity projections to avoid ordering too much or too little material and to hire the right amount of employees to work that productivity.

The dataset will be supplied by the author's work management system and has the following information:

Employee Productivity Data
This data has nine columns with the following information: 

  employee = In the real data this would actually be the employee's name.
  
  reg_hours = This represents the number of regular hours that an employee worked. 
  
  ot_hours = This represents the number of overtime that an employee worked.
  
  vac_hours = This represents the number of vacation hours that an employee took.
  
  task1_hrs = This task represents the amount of time an employee spent on new business work. 
  
  task2_hrs = This task represents the amount of time an employee spent on maintenance work. 
  
  task3_hrs = This task represents the amount of time an employee spent on training.
  
  task4_hrs = This task represents the amount of time an employee spent on emergenecy work.
  
  productivity = This represents the total yearly productivity of that employee.
  
The data product will perform the following functions:
1. Allow the user to upload their dataset to the data product hosted on Streamlit.
2. Cache the data so that every time the user interacts with the data it isn't reuploaded.
3. Output the first five rows of data to have the user verify that the data was uploaded successfully.
4. Output the data types of each variable to identify if variables are categorical or numerical.
5. Output the number of null values found in each variable. 
6. Output a box plot showing the outliers.
7. Allows the user to select the independent and dependent variables by pressing a check box.
8. Scales the data and adds a constant (y-intercept).
9. Applies a multiple linear regression model.
10. Performs a linear relationship assumption test by outputting graphs.
11. Performs a multicollinearity assumption test by outputting a correlation heatmap and VIF values.
12. Performs an independence assumption test by outputting the Durbin-Watson score.
13. Outputs a predictions vs standardized residuals graph to test for homoscedasticity.
14. Performs a multivariate normality assumption test by plotting the Q-Q plot.
15. Outputs the model summary results.
16. Outputs the model's performance metrics (MAE, MSE, and RMSE).

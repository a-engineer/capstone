import pandas as pd
import numpy as np
import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import matplotlib.pyplot as plt
from missingpy import MissForest
from sklearn.preprocessing import OneHotEncoder
from pickle import dump, load

# The following removes the limitations on the maximimum amount of
# row and columns that can be seen. Helps with seeing all the data.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# The following imports the dataset.
df = pd.read_csv("D:/GCU/DSC-590 Data Science Capstone Project/exercise_40_train.csv")

# The following converts the abbreviated days in column x3
df['x3'] = df['x3'].replace(['Mon','Tue','Wed','Thur','Fri','Sat','Sun'], \
	['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

# The following will remove the % symbol in column x7
# and will also convert the column from an 'object' to
# a 'float64' type column.
df['x7'] = df['x7'].str.replace('%','').astype(np.float64)

# The following will remove the $ symbol in column x19
# and will also convert the column from an 'object' to
# a 'float64' type column.
df['x19'] = df['x19'].str.replace('$','').astype(np.float64)

# Replaces 'NA' with 'no' in column x99.
df['x99'] = df['x99'].fillna('no')

#Checking the unique value counts in columns
featureValues={}
for d in df.columns.tolist():
    count=df[d].nunique()
    if count==1:
        featureValues[d]=count
# List of columns having same 1 unique value        
cols_to_drop= list(featureValues.keys())
print("Columns having 1 unique value are : ",cols_to_drop)

# Dropping column x39 since it only has only unique value.
# Columns with one unique value mean a zero variance and won't contribute
# to overall result.
df = df.drop('x39', axis=1)

# Checking to see the percentage of missing values in each column
# Columns with 85% missing value will be dropped
null_val = df.isnull().mean() * 100
print(null_val.sort_values(ascending=False).head(10))

# Code above resulted in x44 with 85.6% null values
df = df.drop('x44',axis=1)

# The following lumps all categorical variables into the 
# cat_var variable.
#cat_var = df.select_dtypes(include=['object']).copy()
#print(cat_var.head())

# Imputes missing categorical values with the mode
df['x24'] = df['x24'].fillna(df['x24'].mode().iloc[0])
df['x33'] = df['x33'].fillna(df['x33'].mode().iloc[0])
df['x77'] = df['x77'].fillna(df['x77'].mode().iloc[0])

# Checks to see there are no more missing values in 
# the categorical variables
print("Null Values for Categorical Variables: ")
cat_var = df.select_dtypes(include=['object']).copy()
print(cat_var.isnull().sum())

# One-Hot Encoding of Categorical Variables
#encoder = OneHotEncoder(handle_unknown='ignore')
#encoder_df = pd.DataFrame(encoder.fit_transform(df[['x3']]).toarray())
#print(encoder_df.head())

# One-Hot Encoding of Categorical Variables (get_dummies method)
df = pd.get_dummies(df, columns=['x3','x24','x31','x33','x60','x65','x77','x93','x99'])
#print("Get Dummies Method Encoding")
#print(df.head())


print("Columns With Null Values.\n")
print(df.isnull().sum().sort_values(ascending=False).head(15))
df_test = df[['x57','x30','x55']].iloc[1:1000,]
print(df_test.head())


# The following performs imputting the null values using MissForest algorithm.
imputer = MissForest()
imputted_df = imputer.fit_transform(df_test)
print("Finished with MissForest Algorithm!")

# https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/


# save the model
dump(imputted_df, open('model.pkl', 'wb'))
# save the scaler
#dump(scaler, open('scaler.pkl', 'wb'))
# load the model
model = load(open('model.pkl', 'rb'))
# load the scaler
#scaler = load(open('scaler.pkl', 'rb'))

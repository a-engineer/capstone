import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression



X, y, lin_coef = make_regression(n_samples=150, n_features=6, n_informative = 6, random_state=1, n_targets = 1,noise=0.2, coef = True)
X_data = np.array(X)
y_data = np.array(y)
sc = StandardScaler()
X_sc = sc.fit_transform(X_data)
new_data = np.append(X_sc,y_data.reshape(-1,1),axis=1)
df = pd.DataFrame(new_data, columns=['x1','x2','x3','x4','x5','x6','productivity'])
print(df.head())
df.to_csv("make_regression_sample_output.csv", index = False)
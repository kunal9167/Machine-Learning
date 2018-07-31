# -*- coding: utf-8 -*-
"""
Created on Tue May 29 01:25:56 2018

@author: Kunal Gupta
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Splitting the dataset into Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# Feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#Fitting the Regression model to the dataset



# Predicting a new result with Polynomial Regression
y_pred = regressor.predict(6.5)

# Visualise the Polynomial Regression Results
plt.scatter(x,y, color = 'red')
plt.plot(x, regressor.predict(x),color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualise the Polynomial Regression Results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x),0.1)
X_grid.reshape(len(X_grid),1)
plt.scatter(x,y, color = 'red's)
plt.plot(X_grid, regressor.predict(X_grid),color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()





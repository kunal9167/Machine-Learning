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
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

# Feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualise the linear Regression Results
#plt.scatter(x,y, color = 'red')
#plt.plot(x, lin_reg.predict(x),color = 'blue')
#plt.title("Truth or Bluff (Linear Regression)")
#plt.xlabel("Position Level")
#plt.ylabel("Salary")
#plt.show()

# Visualise the Polynomial Regression Results
X_grid = np.arange(min(x),max(x),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))




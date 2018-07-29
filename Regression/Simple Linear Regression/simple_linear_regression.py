# -*- coding: utf-8 -*-
"""
Created on Tue May 29 01:25:56 2018

@author: Kunal Gupta
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1:2].values

# Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

# Feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(X_train)

# Visualizing the training set results
plt.scatter(X_train, y_train,color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Training set)")
plt.ylabel("Years of Experience")
plt.show()

# Visualizing the training set results
plt.scatter(X_test, y_test,color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Training set)")
plt.ylabel("Years of Experience")
plt.show()












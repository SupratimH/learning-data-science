"""
Created on Tue Sep 15 2018
@author: Supratim Haldar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import logisticRegression
import time

# Load training data. This dataset is downloaded from
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# The other training data is from Andrew NG's M/L course
def loadTrainingData():
    #train_data = pd.read_csv("C:\Study\DataSets\House_Prices-Advanced_Regression_Techniques\\train.csv")
    train_data = pd.read_csv('C:\Study\Data Science\Coursera Machine Learning - Andrew NG\Assignments\machine-learning-ex2\ex2\ex2data1.csv')
    #print(train_data.isnull().sum())
    return train_data

# Normalise training data with mean and standard deviation
def normaliseFeature(train_data, feature = None):
    if feature:
        mean = train_data[feature].mean()
        std = train_data[feature].std()
        train_data[feature] = (train_data[feature] - mean)/std
    else:
        train_data = (train_data - train_data.mean())/train_data.std()
    return train_data


# Test my vectorized implementations of linear regression algorithm
def test_LinearRegression_Vectorized(train_data):
    # Total number of records
    m = len(train_data)
    
    # Populate y data into a m-dim vector
    # And then drop that column from feature list
    data_y = train_data.Admission.values.reshape(m, 1)
    train_data = train_data.drop('Admission', 1) 
    
    # Copy selected features
    train_data = train_data.loc[:,['Exam1','Exam2']]

    # Normalise features  
    #train_data = train_data.apply(normaliseFeature, axis = "rows")

    # Setting first feature to 1, this is the bias/y-intercept or theta0
    train_data['first_dummy_feature'] = 1 
    train_data = train_data.loc[:,['first_dummy_feature', 'Exam1','Exam2']]

    # Populate X (features) data into a mxn matrix
    data_X = train_data.values
    
    # Set initial theta to 0's in a n-dim vector
    n = len(data_X[0])
    initial_theta = np.zeros([n, 1])
    #initial_theta = np.array([[-24],[0.2],[0.2]])

    # Calculate logistic regression cost with initial_theta of 0
    #np.set_printoptions(precision=4, suppress=True)
    cost = logisticRegression.computeCost(data_X, data_y, initial_theta)
    print("Logistic Regression: Cost with 0 Theta (Vectorized) = ", cost)
    
    # Perform gradient descent to find optimal theta
    alpha, num_iters = 0.001, 1000
    theta, J_history = logisticRegression.gradientDescent(data_X, data_y, \
                initial_theta, alpha, num_iters)
    print("Logistic Regression: Theta after Gradient Descent (Vectorized) = ", np.around(theta, 4))
    
    # Calculate linear regression cost with theta after gradient descent
    cost = logisticRegression.computeCost(data_X, data_y, theta)
    print("Logistic Regression: Cost after GD (Vectorized) = ", cost)
    
    pyplot.plot(J_history)

    
def main():
    start_time = time.time()
    train_data = loadTrainingData()
    #test_LinearRegression(train_data)
    test_LinearRegression_Vectorized(train_data)
    print("Execution time in Seconds =", time.time() - start_time)

if __name__ == '__main__':
    main()

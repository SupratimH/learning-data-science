# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 2018
@author: Supratim Haldar
"""

import pandas as pd
import matplotlib.pyplot as pyplot
import linearRegression

# Load training data. This dataset is downloaded from
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
def loadTrainingData():
    #train_data = pd.read_csv("C:\Study\DataSets\House_Prices-Advanced_Regression_Techniques\\train.csv")
    train_data = pd.read_csv('C:\Study\Data Science\Coursera Machine Learning - Andrew NG\Assignments\machine-learning-ex1\ex1\ex1data2.csv')
    #print(train_data.isnull().sum())
    return train_data


def normaliseFeature(train_data, feature = None):
    if feature:
        mean = train_data[feature].mean()
        std = train_data[feature].std()
        train_data[feature] = (train_data[feature] - mean)/std
    else:
        train_data = (train_data - train_data.mean())/train_data.std()
    return train_data


# Test my implementations of linear regression algorithm
def test_LinearRegression(train_data):
    # Populate y data and then drop the column from feature list
    data_y = train_data.SalePrice
    train_data = train_data.drop('SalePrice', 1) 
    
    # Normalise features  
    #train_data = normaliseFeature(train_data, 'LotArea')
    #train_data = normaliseFeature(train_data, 'YearBuilt')
    #train_data = normaliseFeature(train_data, 'GrLivArea')
    #train_data = normaliseFeature(train_data, 'BedroomAbvGr')
    train_data = train_data.apply(normaliseFeature, axis = "rows")
    
    #Setting first feature to 1, this is the bias/intercept or theta0
    train_data['first_dummy_feature'] = 1 
    train_data_list = []

    # Populate features row-by-row in a list of list
    for i in range(len(train_data)):
        train_data_list.append(list(train_data.loc[i,['first_dummy_feature', \
            'LotArea','BedroomAbvGr']]))
            #'LotArea','YearBuilt','GrLivArea','BedroomAbvGr']]))

    # Populate X (features) data
    data_X = train_data_list
    initial_theta = [0] * len(data_X[0])
    
    # Calculate linear regression cost with initial_theta of 0
    cost = linearRegression.computeCostMulti(data_X, data_y, initial_theta)
    print("Linear Regression: Cost with 0 Theta = ", cost)
    
    # Perform gradient descent to find optimal theta
    alpha, num_iters = 0.01, 400
    theta, J_history = linearRegression.gradientDescentMulti(data_X, data_y, \
                initial_theta, alpha, num_iters)
    print("Linear Regression: Theta after Gradient Descent = ", theta)
    
    # Calculate linear regression cost with theta after gradient descent
    cost = linearRegression.computeCostMulti(data_X, data_y, theta)
    print("Linear Regression: Cost after GD = ", cost)
    
    pyplot.plot(J_history)
    
    
def main():
    train_data = loadTrainingData()
    test_LinearRegression(train_data)

if __name__ == '__main__':
    main()

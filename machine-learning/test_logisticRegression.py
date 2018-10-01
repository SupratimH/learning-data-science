"""
Created on Tue Sep 15 2018
@author: Supratim Haldar
"""

import pandas as pd
import numpy as np
from scipy import optimize
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as pyplot
import logisticRegression
import time
import imageio

# Load training data
def loadTrainingData(path):
    train_data = pd.read_csv(path)
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

# Predict digit from image - Just for Fun
def predictDigitFromImage(theta):
    im = imageio.imread('C:\Study\DataSets\MNIST_Handwritten_Digit_Recognizer\\3.png', as_gray=True)
    data_im = np.ones([im.flatten().shape[0]+1])
    data_im[1:] = im.flatten()
    data_im = data_im.reshape(1, data_im.shape[0])
    print("Predict Digit: Input data =", im.shape, data_im.shape, theta)

    Z = logisticRegression.sigmoid(np.dot(data_im, theta.T))
    prediction = np.argmax(Z, axis=1)
    prob_max = np.max(Z, axis=1)
    print("Predict Digit: Prediction Result and Probability =", Z, prediction, prob_max)
    

# Test my vectorized implementations of logistic regression algorithm
def test_LogisticRegression_Vectorized():
    # This dataset is downloaded from Andrew NG's M/L course
    train_data = loadTrainingData('C:\Study\Data Science\Coursera Machine Learning - Andrew NG\Assignments\machine-learning-ex2\ex2\ex2data1.csv')

    # Total number of records
    m = len(train_data)
    
    # Populate y data into a m-dim vector
    # And then drop that column from feature list
    data_y = train_data.Admission.values.reshape(m, 1)
    train_data = train_data.drop('Admission', 1) 
    
    # Copy selected features
    train_data = train_data.loc[:,['Exam1','Exam2']]

    # ==== Normalise features =====
    #train_data = train_data.apply(normaliseFeature, axis = "rows")

    # Setting first feature to 1, this is the bias/y-intercept or theta0
    train_data['first_dummy_feature'] = 1 
    train_data = train_data.loc[:,['first_dummy_feature', 'Exam1','Exam2']]

    # Populate X (features) data into a mxn matrix
    data_X = train_data.values
    
    # Set initial theta to 0's in a n-dim vector
    n = len(data_X[0])
    initial_theta = np.zeros([n, 1])

    # Calculate logistic regression cost with initial_theta of 0
    np.set_printoptions(precision=4, suppress=True)
    cost = logisticRegression.computeCost(initial_theta, data_X, data_y)
    print("Logistic Regression: Cost with 0 Theta = ", cost)
    
    # Perform gradient descent to find optimal theta
    alpha, num_iters, lambda_reg = 0.001, 100000, 2
    theta, theta_grad, J_history = logisticRegression.gradientDescent(initial_theta, \
                data_X, data_y, alpha, num_iters, lambda_reg)
    print("Logistic Regression: Theta after Gradient Descent = ", theta)
    print("Logistic Regression: Final Derivative of Theta after Gradient Descent = ", theta_grad)
    
    # Calculate linear regression cost with theta after gradient descent
    cost = logisticRegression.computeCost(theta, data_X, data_y, lambda_reg)
    print("Logistic Regression: Cost after GD (Vectorized) = ", cost)
    
    # Predict result on input dataset
    pred, accuracy = logisticRegression.predict(theta, data_X, data_y)
    
    #pyplot.plot(J_history)
    
    # Calculate parameter values using advanced optimization algorithms
#    def local_computeGradient(initial_theta):
#        return logisticRegression.computeGradient(initial_theta, data_X, data_y, lambda_reg)
#    theta_optimized = fmin_bfgs( \
#            local_computeGradient, \
#            initial_theta \
#            )
    theta_optimized = optimize.minimize( \
        logisticRegression.computeGradient, \
        initial_theta, \
        args=(data_X, data_y, lambda_reg), \
        method = "BFGS", 
        jac=True, options={'disp': True} \
        )
    print("Logistic Regression: Theta after Advanced Optimization = ", theta_optimized)

    # Calculate linear regression cost with theta after gradient descent
    cost = logisticRegression.computeCost(theta_optimized.x.reshape(n,1), data_X, data_y, lambda_reg)
    print("Logistic Regression: Cost after Advanced Optimization  = ", cost)
    
    # Predict result on input dataset
    pred, accuracy = logisticRegression.predict(theta_optimized.x.reshape(n,1), data_X, data_y)


# Test my implementation of one-vs-all logistic regression algorithm
def test_OneVsAll():
    # This dataset is downloaded from Kaggle
    train_data = loadTrainingData('C:\Study\DataSets\MNIST_Handwritten_Digit_Recognizer\\train.csv')

    # Total number of records
    m = len(train_data)
    
    # Populate y data into a m-dim vector
    # And then drop that column from feature list
    num_labels = len(train_data.label.unique())
    data_y = train_data.label.values.reshape(m, 1)
    train_data = train_data.drop('label', 1)
    
    # Setting first feature to 1, this is the bias/y-intercept or theta0
    train_data.insert(0, 'first_dummy_feature', 1)

    # Populate X (features) data into a mxn matrix
    data_X = train_data.values
    
    # Reduce the training data set
    data_X_1 = data_X[15000:20000, :]
    data_y_1 = data_y[15000:20000, :]
    
    # Call one-vs-all calculation
    lambda_reg = 1
    all_theta = logisticRegression.oneVsAll(data_X_1, data_y_1, num_labels, lambda_reg)
    print("OneVsAll: Theta after Advanced Optimization =", all_theta.shape)
    
#    # Predict results of test data (on test data from Kaggle dataset)
#    test_data = loadTrainingData('C:\Study\DataSets\MNIST_Handwritten_Digit_Recognizer\\test.csv')
#    test_data_m = len(test_data)
#    test_data.insert(0, 'first_dummy_feature', 1)
#    test_data_X = test_data.values
    
    # Predict results of test data (from subset of input training data)
    test_data_X = data_X[25000:30000, :]
    test_data_y = data_y[25000:30000, :]
    
    Z = logisticRegression.sigmoid(np.dot(test_data_X, all_theta.T))
    prediction = np.argmax(Z, axis=1)
    prob_max = np.max(Z, axis=1)
    print("OneVsAll: Prediction Result =", prediction.shape)
    accuracy = np.mean(prediction.reshape(test_data_y.shape) == test_data_y) * 100
    print("OneVsAll: Prediction Accuracy % =", accuracy)
    
    # Predict the digit I have written from the PNG file supplied
    predictDigitFromImage(all_theta)
    
#    # Prepare submission file
#    my_submission = pd.DataFrame({ \
#            'ImageId': np.arange(1, data_X_1.shape[0]+1), \
#            'Actual Label': data_y_1.flatten(), \
#            'Probability': prob_max.flatten(), \
#            'Label': prediction.flatten()})
#    my_submission.to_csv('C:\Study\DataSets\MNIST_Handwritten_Digit_Recognizer\\SH_submission.csv', index=False)
    
    
def main():
    start_time = time.time()
    #test_LogisticRegression_Vectorized()
    test_OneVsAll()
    print("Execution time in Seconds =", time.time() - start_time)

if __name__ == '__main__':
    main()

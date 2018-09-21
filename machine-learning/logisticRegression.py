"""
Created on Tue Sep 14 2018
@author: Supratim Haldar
@Description: My implementation of logistic regression (classifier) algorithm
"""
import numpy as np

# =============================================================================
# Function to calculate value Sigmoid Function of any variable z.
# z can be a matrix, vector or scalar
# sigmoid g(z) = 1/(1 + e^-z)
# =============================================================================
def sigmoid(z):
    sig = 1.0/(1.0 + np.exp(-z))
    
    # Due to floating point presision related issues, e^-z might return very 
    # small or very large values, resulting in sigmoid = 1 or 0. Since we will
    # compute log of these values later in cost function, we want to avoid 
    # sig = 1 or 0, and hardcode to following values instead.
    sig[sig == 1.0] = 0.9999
    sig[sig == 0.0] = 0.0001
    
    return sig


# =============================================================================
# Compute cost of Logistic Regression with multiple features
# Vectorized implementation
# Input: data_X = mxn matrix, data_y = m-dim vector, theta = n-dim vector
# Output: cost = 1-dim vector
# =============================================================================
def computeCost(data_X, data_y, theta):
    # No of rows
    m = len(data_X)
    
    # h(x) = g(z) = g(theta0 + theta1*X1 + theta2*X2 + .. + thetan*Xn)
    # h(x) = g(X * theta) = Sigmoid(X * theta) = m-dim vector
    hx = sigmoid(np.dot(data_X, theta))
    cost = - np.dot(data_y.T, np.log(hx)) - np.dot((1 - data_y).T, np.log(1 - hx))
    J = cost/m
    return J
    

# =============================================================================
# Gradient Descent of Linear Regression with multiple features
# Vectorized implementation
# Input: data_X = mxn matrix, data_y = m-dim vector, theta = n-dim vector
# alpha = learning rate, num_iters = no of iterations/steps for GD
# Output: theta = n-dim vector, 
# J_history = cost at each iteration, a num_iters-dim vector
# =============================================================================
def gradientDescent(data_X, data_y, theta, alpha, num_iters):
    m = len(data_X) # No of rows
    J_history = np.zeros([num_iters, 1])

#    hx = sigmoid(np.dot(data_X, theta))
#    error = hx - data_y
#    theta = (np.dot(data_X.T, error))/m
    
    for i in range(num_iters):
        hx = np.zeros(data_y.shape)
        error = np.zeros(data_y.shape)
        theta_change = np.zeros(theta.shape)
        
        hx = sigmoid(np.dot(data_X, theta))
        error = hx - data_y
        theta_change = (alpha) * (np.dot(data_X.T, error)/m)
        theta = theta - theta_change
        
        J_history[i] = computeCost(data_X, data_y, theta)
        #print("J_history", i, J_history[i])
        
    return theta, J_history


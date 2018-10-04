"""
Created on Tue Oct 04 2018
@author: Supratim Haldar
@Description: This is an attempt to understand the linear regression algorithm
and gradient computation in an intuitive way.
"""
import numpy as np
import matplotlib.pyplot as pyplot
import time
import seaborn as sns

# =============================================================================
# Computation of cost or objective
# This is a local copy of the GD functions implemented in linearRegression.py
# =============================================================================
def computeCostMulti(data_X, data_y, theta):
    # No of rows
    m = len(data_X)
    
    # h(x) = theta0 + theta1*X1 + theta2*X2 + .. + thetan*Xn
    # hx = X * theta = m-dim vector
    hx = np.dot(data_X, theta)
    error = hx - data_y
    error_sq = np.power(error, 2)
    cost = sum(error_sq/(2*m))
    
    cost.astype(float, copy=False)
    return np.around(cost, decimals=4)


# =============================================================================
# Computation of gradient
# This is a local copy of the GD functions implemented in linearRegression.py
# =============================================================================
def gradientDescent(data_X, data_y, theta, alpha, num_iters):
    m = len(data_X) # No of rows
    J_history = np.zeros([num_iters, 1])
    theta_history = np.zeros([num_iters, 1])
    
    for i in range(num_iters):
        hx = np.dot(data_X, theta)
        error = hx - data_y
        theta_change = (alpha/m) * np.dot(data_X.T, error)
        theta = theta - theta_change
        
        J_history[i] = computeCostMulti(data_X, data_y, theta)
        theta_history[i] = theta
        
    return theta, J_history, theta_history


# =============================================================================
# y = function of x
# =============================================================================
def f(x):
    return 5*x


# =============================================================================
# Trying to build an intuitative understanding of linear regression
# =============================================================================
def understand_LogisticRegression():
    m = 1000
    x = np.arange(m).reshape([m,1])
    y = f(x).reshape([m,1])
    #print("x =", x, "y =", y)
    #pyplot.scatter(x, y)
    alpha = 0.0000001
    num_iters = 100
    initial_theta = np.zeros([1,1])
    theta, J_History, theta_history = gradientDescent(x, y, initial_theta, alpha, num_iters)
    print("Computed Theta =", theta_history)
    pyplot.scatter(theta_history, J_History, marker='+')
    

def main():
    start_time = time.time()
    understand_LogisticRegression()
    print("Execution time in Seconds =", time.time() - start_time)

if __name__ == '__main__':
    main()
"""
Created on Tue Sep 11 2018
@author: Supratim Haldar
@Description: My implementation of linear regression algorithm
"""
import numpy as np

# =============================================================================
# Compute cost of Linear Regression with single feature. Normal implementation.
# Also called Squared Error Function or Mean Squared Error.
# J(theta) = (1/2m)SUM((h(x) - y))^2
# Input: List of X, y and theta(params)
# Output: Cost value
# =============================================================================
def computeCost(data_X, data_y, theta):
    m = len(data_X) # No of records
    error_sq, cost = 0, 0
    
    for i in range(0,m):
        hx = theta[0] + theta[1]*data_X[i] # h(x) = theta0 + theta1*X
        error_sq = error_sq + (hx - data_y[i])**2
        
    cost = error_sq/(2*m)
    return cost        


# =============================================================================
# Compute cost of Linear Regression with multiple features
# Normal implementation
# =============================================================================
def computeCostMulti(data_X, data_y, theta):
    m = len(data_X) # No of records
    n = len(data_X[0]) # No of features
    error_sq, cost = 0, 0
    hx = [0] * m
    
    for i in range(0,m):
        for j in range(0,n):
            hx[i] = hx[i] + theta[j] * data_X[i][j]
        error_sq = error_sq + (hx[i] - data_y[i])**2
        
    cost = (error_sq)/(2*m)
    return round(cost, 4)


# =============================================================================
# Compute cost of Linear Regression with multiple features
# Vectorized implementation
# Input: data_X = mxn matrix, data_y = m-dim vector, theta = n-dim vector
# Output: cost = 1-dim vector
# =============================================================================
def computeCostMulti_Vectorized(data_X, data_y, theta):
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
# Perform gradient descent on Linear Regression with multiple features
# Normal implementation
# =============================================================================
def gradientDescentMulti(data_X, data_y, theta, alpha, num_iters):
    m = len(data_X) # No of records
    n = len(data_X[0]) # No of features
    J_history = [0] * num_iters
    
    for iteration in range(num_iters):
        hx = 0
        error = [0.5] * m
        cost = 0
        i, j, k, l = 0, 0, 0, 0
        #print("Iteration no =", iteration, theta)
        
        for i in range(m): # Loop on rows
            hx = 0
            for j in range(n): # Loop on features of each rows
                hx = float(hx + (theta[j] * data_X[i][j]))
            error[i] = float(hx - data_y[i])
    
        for k in range(n): # Loop on features of each rows
            cost = 0
            for l in range(m): # Loop on rows
                cost = float(cost + (error[l] * data_X[l][k]))
            theta[k] = float(theta[k] - (alpha*cost)/m)

        # Calculate linear regression cost with new theta after gradient descent
        J_history[iteration] = float(computeCostMulti(data_X, data_y, theta))
        #print('Cost = %0.4f' % J_history[iteration])
            
    return theta, J_history


# =============================================================================
# Gradient Descent of Linear Regression with multiple features
# Vectorized implementation
# Input: data_X = mxn matrix, data_y = m-dim vector, theta = n-dim vector
# alpha = learning rate, num_iters = no of iterations/steps for GD
# Output: theta = n-dim vector, 
# J_history = cost at each iteration, a num_iters-dim vector
# =============================================================================
def gradientDescentMulti_Vectorized(data_X, data_y, theta, alpha, num_iters):
    m = len(data_X) # No of rows
    J_history = np.zeros([num_iters, 1])
    
    for i in range(num_iters):
        hx = np.dot(data_X, theta)
        error = hx - data_y
        theta_change = (alpha/m) * np.dot(data_X.T, error)
        theta = theta - theta_change
        
        J_history[i] = computeCostMulti_Vectorized(data_X, data_y, theta)
        
    return theta, J_history


            
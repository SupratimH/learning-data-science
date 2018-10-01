"""
Created on Tue Sep 14 2018
@author: Supratim Haldar
@Description: My implementation of logistic regression (classifier) algorithm
"""
import numpy as np
from scipy import optimize

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
def computeCost(theta, data_X, data_y, lambda_reg = 0):
    m = len(data_X) # No of rows
    n = len(data_X[0]) # No of features
    theta = theta.reshape(n,1)
    
    # h(x) = g(z) = g(theta0 + theta1*X1 + theta2*X2 + .. + thetan*Xn)
    # h(x) = g(X * theta) = Sigmoid(X * theta) = m-dim vector
    hx = sigmoid(np.dot(data_X, theta))
    cost = - np.dot(data_y.T, np.log(hx)) - np.dot((1 - data_y).T, np.log(1 - hx))
    
    # This is unregularized cost
    J = cost/m
    
    # Adding regularization. Setting theta0 to 0, because theta0 will not be 
    # regularized
    J_reg = (lambda_reg/(2*m)) * np.dot(theta[1:,:].T, theta[1:,:])
    J = J + J_reg
    
    return J
    

# =============================================================================
# Compute gradient or derivative of cost function over parameter, i.e.
# d J(Theta)/d Theta
# =============================================================================
def computeGradient(theta, data_X, data_y, lambda_reg = 0):
    m = len(data_X) # No of rows
    n = len(data_X[0]) # No of features
    theta = theta.reshape(n,1)
    theta_gradient = np.zeros(theta.shape)
    cost = 0
    #print("==== Inside computeGradient() ====", data_X.shape, data_y.shape)

    cost = computeCost(theta, data_X, data_y, lambda_reg)
    
    hx = sigmoid(np.dot(data_X, theta))
    error = hx - data_y
    theta_gradient = (1/m) * (np.dot(data_X.T, error))
    
    # Apply regularization
    theta_reg = (lambda_reg/m) * theta[1:,:]
    theta_gradient[1:,:] = theta_gradient[1:,:] + theta_reg
    
    #print("==== Inside computeGradient() ====", cost)
    return cost.flatten(), theta_gradient.flatten()


# =============================================================================
# Gradient Descent of Linear Regression with multiple features
# Vectorized implementation
# Input: data_X = mxn matrix, data_y = m-dim vector, theta = n-dim vector
# alpha = learning rate, num_iters = no of iterations/steps for GD
# Output: theta = n-dim vector, 
# J_history = cost at each iteration, a num_iters-dim vector
# =============================================================================
def gradientDescent(theta, data_X, data_y, alpha, num_iters, lambda_reg = 0):
    m = len(data_X) # No of rows
    J_history = np.zeros([num_iters, 1])

    for i in range(num_iters):
        hx = np.zeros(data_y.shape)
        error = np.zeros(data_y.shape)
        theta_change = np.zeros(theta.shape)
        
        hx = sigmoid(np.dot(data_X, theta))
        error = hx - data_y
        theta_change = (alpha/m) * (np.dot(data_X.T, error))
        
        # Apply regularization
        temp = theta[0,0]
        theta[0,0] = 0
        theta_reg = (lambda_reg/m) * theta
        theta[0,0] = temp;
        theta_change = theta_change + theta_reg
        
        theta = theta - theta_change
        
        J_history[i] = computeCost(theta, data_X, data_y, lambda_reg)
        
    return theta, theta_change, J_history

# =============================================================================
# Predict results based on test input feature and parameter values
# Compare with output results, if already available
# =============================================================================
def predict(theta, data_X, data_y):
    prob = sigmoid(np.dot(data_X, theta))
    pred = prob >= 0.5
    accuracy = np.mean((pred == data_y)) * 100
    print("Predict: Prediction Accuracy % =", accuracy)
    return pred, accuracy


# =============================================================================
# One vs All method of logistic regression
# Used for data with multiple clssification outputs
# =============================================================================
def oneVsAll(data_X, data_y, num_labels, lambda_reg):
    n = data_X.shape[1] # No of features
    all_theta = np.zeros([num_labels, n])
    initial_theta = np.zeros([n, 1])
    print("OneVsAll: Shape of X and y: ", data_X.shape, data_y.shape)
    
    for label in range(num_labels):
        # Calling advanced optimization alogorith to converge gradient
        theta_optimized = optimize.minimize( \
            computeGradient, \
            initial_theta, \
            args=(data_X, data_y == label, lambda_reg), \
            method = "CG", 
            jac=True, options={'disp': True, 'maxiter': 100} \
            )
        print("OneVsAll: Optimization Result =", theta_optimized.message, theta_optimized.success)
        theta = theta_optimized.x.reshape(n, 1)
        all_theta[label,:] = theta.T

    return all_theta


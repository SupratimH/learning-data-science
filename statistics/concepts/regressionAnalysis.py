# -*- coding: utf-8 -*-
"""
Created on Tue Sep 02 2018
@author: Supratim Haldar
"""
import math
import summaryStatistics as ss
from functools import reduce

""" Calculate Correlation Coefficient, r = 1/(n-1)*Sum(Z_X * Z_Y) """
def CorrCoeff(data_X, data_Y):
    nX, nY = len(data_X), len(data_Y)
    if (nX != nY):
        print("Length of X and Y datasets do not match!")
        return -1
    
    result = 0
    sd_X, sd_Y = ss.StdDev(data_X), ss.StdDev(data_Y)
    mean_X, mean_Y = ss.Mean(data_X), ss.Mean(data_Y)

    for i in range(0, nX, 1):
        result = result + (((data_X[i] - mean_X)/sd_X) * ((data_Y[i] - mean_Y)/sd_Y))

    r = result/(nX - 1)
    return r


""" Calculate Pearson Correlation Coefficient """
def PearsonCorrCoeff(data_X, data_Y):
    nX, nY = len(data_X), len(data_Y)
    assert nX == nY and nX != 0, "Length of X and Y datasets do not match!"
    n = nX
        
    # Traditional implementation with for loop
#    sum_X = sum_Y = sum_XY = sum_X2 = sum_Y2 = 0
#    for i in range(0, n, 1):
#        sum_X = sum_X + data_X[i]
#        sum_Y = sum_Y + data_Y[i]
#        sum_XY = sum_XY + (data_X[i] * data_Y[i])
#        sum_X_sq = sum_X2 + pow(data_X[i], 2)
#        sum_Y_sq = sum_Y2 + pow(data_Y[i], 2)
        
    # Declarative style implementation
    sum_X = reduce(lambda a,b: a+b, data_X)
    sum_Y = reduce(lambda a,b: a+b, data_Y)
    sum_XY = sum(map(lambda x,y: x*y, data_X, data_Y))
    sum_X_sq = sum(map(lambda a: a**2, data_X))
    sum_Y_sq = sum(map(lambda a: a**2, data_Y))

    result_Num = float((n*sum_XY - sum_X*sum_Y))
    result_Den = float(n*sum_X_sq - pow(sum_X, 2)) * float(n*sum_Y_sq - pow(sum_Y, 2))
    result = float(result_Num)/float(math.sqrt(result_Den))

    return result

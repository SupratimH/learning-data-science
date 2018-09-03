# -*- coding: utf-8 -*-
"""
Created on Tue Sep 02 2018
@author: Supratim Haldar
"""

import pandas as pd
import matplotlib.pyplot as pyplot
import regressionAnalysis as ra
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

# Load training data. This dataset is downloaded from
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
def loadTrainingData():
    train_data = pd.read_csv("C:\Study\DataSets\House_Prices-Advanced_Regression_Techniques\\train.csv")
    return train_data

# Test my implementations of correlation evaluations
def test_Correlation(train_data):
    data_X = train_data.LotArea
    data_Y = train_data.SalePrice
    
    pyplot.scatter(data_X, data_Y, alpha=0.5)
    
    # Calculate corr coeff
    corrCoeff = ra.CorrCoeff(data_X, data_Y)
    print("Correlation Coefficient = ", round(corrCoeff, 4))
    
    # Calculate Pearson's corr coeff
    pearsonCorrCoeff = ra.PearsonCorrCoeff(data_X, data_Y)
    print("Pearson Correlation Coefficient = ", round(pearsonCorrCoeff, 4))
    
    # Calculate Pearson's corr coeff using SciPy library to validate the result
    print("Pearson Correlation Coefficient from SciPy = ", round(pearsonr(data_X, data_Y)[0], 4))

    # Calculate Spearman's corr coeff
    spearmanCorrCoeff = ra.SpearmanCorrCoeff(data_X, data_Y)
    print("Spearman Correlation Coefficient = ", round(spearmanCorrCoeff, 4))
    
    # Calculate Spearman's corr coeff using SciPy library to validate the result
    print("Spearman Correlation Coefficient from SciPy = ", round(spearmanr(data_X, data_Y)[0], 4))

    # Calculate Covariance
    cov = ra.Covariance(data_X, data_Y)
    print("Covariance = ", round(cov, 4))
    
    
def main():
    train_data = loadTrainingData()
    test_Correlation(train_data)

if __name__ == '__main__':
    main()

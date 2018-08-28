# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:35:17 2018

@author: SUHALDAR
"""
import math
from operator import itemgetter

""" Function to calculate Mean """
def CalculateMean(data):
    return float(sum(data)/len(data))

""" Function to calculate Median """
def CalculateMedian(data):
    sortedData = sorted(data) 
    length = len(data)
    
    if length%2 == 0:
        median = (sortedData[length//2 - 1] + sortedData[length//2])/2
    else:
        median = sortedData[length//2]

    return median

""" Function to calculate Mode """
def CalculateMode(data, type='max'):
    dict_Data = {}
    for rec in data:
        dict_Data[rec] = dict_Data.get(rec, 0) + 1

    if type == 'max':
        modeData = sorted(dict_Data.items(), key=itemgetter(1))[-1]
        return modeData[0], modeData[1]
    else:
        return dict_Data.keys(), dict_Data.values()

""" Function to calculate Variance """
def CalculateVariance(data):
    mean = CalculateMean(data)
    dev = 0
    for rec in data:
        dev = dev + pow((rec - mean),2)
    
    return dev/len(data)
    
""" Function to calculate Standard Deviation """
def CalculateStdDev(data):
    return math.sqrt(CalculateVariance(data))


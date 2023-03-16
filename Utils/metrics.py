#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:49:55 2022

@author: ben
"""

import numpy as np

class Loss():
    def __init__(self):
        pass
    def loss(self):
        print('loss')
        
def MeanSquaredError(y_true, y_pred):
    mse = np.sum((y_true - y_pred)**2)/(2*len(y_true))
    return mse
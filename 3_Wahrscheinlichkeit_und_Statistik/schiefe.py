#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ben
"""

import numpy as np

def mean(x):
    mean=0
    for xi in x:
        mean+=xi
    return mean/len(x)
def std(x):
    std=0
    x_mean = mean(x)
    for xi in x:
        std+=(xi-x_mean)**2
    return np.sqrt(std/(len(x)))
def skew(x):
    x_mean = mean(x)
    x_std = std(x)
    skew = ((x-x_mean)/x_std)**3
    return mean(skew)

if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y = 1/(np.sqrt(4*np.pi)) * np.exp( -0.2*(x)**2  )
    print('Skewness for the input data : ', skew(y))
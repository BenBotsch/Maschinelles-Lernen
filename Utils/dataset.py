#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:23:05 2022

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt

def arctan_fun():
    n_samples = 100
    x = np.sort(np.random.randint(500, size=n_samples))/100
    y = np.arctan(x)
    y = y + np.random.randn(n_samples) * 0.1
    return x.reshape(-1,1),y.reshape(-1,1)

def two_class_example():
    n_samples = 100
    x1_center = 1.0
    x2_center = 3.0
    x1 = x1_center+np.random.normal(0, 0.5, n_samples).reshape(-1,1)
    x2 = x2_center+np.random.normal(0, 0.8, n_samples).reshape(-1,1)
    y = np.zeros((n_samples,1))
    
    x1_center = 2.0
    x2_center = 1.0
    x1 = np.concatenate((x1, x1_center+np.random.normal(0, 0.3, 100).reshape(-1,1)), axis=0)
    x2 = np.concatenate((x2, x2_center+np.random.normal(0, 1.8, 100).reshape(-1,1)), axis=0)
    
    y = np.concatenate((y, np.ones((n_samples,1))), axis=0)
    return np.concatenate((x1,x2), axis=1),y

def three_class_example():
    n_samples = 100
    x1_center = 1.0
    x2_center = 3.0
    x1 = x1_center+np.random.normal(0, 0.5, n_samples).reshape(-1,1)
    x2 = x2_center+np.random.normal(0, 0.8, n_samples).reshape(-1,1)
    y = np.zeros((n_samples,1))
    
    x1_center = 2.0
    x2_center = 1.0
    x1 = np.concatenate((x1, x1_center+np.random.normal(0, 0.3, 100).reshape(-1,1)), axis=0)
    x2 = np.concatenate((x2, x2_center+np.random.normal(0, 1.3, 100).reshape(-1,1)), axis=0)
    
    y = np.concatenate((y, np.ones((n_samples,1))), axis=0)
    
    x1_center = 3.0
    x2_center = 3.0
    x1 = np.concatenate((x1, x1_center+np.random.normal(0, 0.4, 100).reshape(-1,1)), axis=0)
    x2 = np.concatenate((x2, x2_center+np.random.normal(0, 0.6, 100).reshape(-1,1)), axis=0)
    
    y = np.concatenate((y, np.ones((n_samples,1))*2), axis=0)
    return np.concatenate((x1,x2), axis=1),y




if __name__ == "__main__":
    show_plot=2
    if(show_plot==0):
        x,y = arctan_fun()
        plt.plot(x,y)
    elif(show_plot==1):
        x,y = two_class_example()
        
        cm = 1/2.54  # centimeters in inches
        plt.figure(figsize=((4.5*cm, 4*cm)),dpi=600)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.grid(True)
        plt.plot(x[0:int(x.shape[0]/2),0],x[0:int(x.shape[0]/2),1],'b.')
        plt.plot(x[int(x.shape[0]/2):,0],x[int(x.shape[0]/2):,1],'g.')
        plt.xlim((0,3))
        plt.ylim((0,4.5))
        plt.xlabel("x$_1$")
        plt.ylabel("x$_2$")
        plt.savefig('image.pdf',bbox_inches='tight')
        plt.show()
    elif(show_plot==2):
        x,y = three_class_example()
        
        cm = 1/2.54  # centimeters in inches
        plt.figure(figsize=((4.5*cm, 4*cm)),dpi=600)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.grid(True)
        plt.plot(x[0:int(x.shape[0]/3),0],x[0:int(x.shape[0]/3),1],'b.')
        plt.plot(x[int(x.shape[0]/3):int(x.shape[0]*2/3),0],x[int(x.shape[0]/3):int(x.shape[0]*2/3),1],'g.')
        plt.plot(x[int(x.shape[0]*2/3):,0],x[int(x.shape[0]*2/3):,1],'c.')
        plt.xlim((0,4))
        plt.ylim((0,5))
        plt.xlabel("x$_1$")
        plt.ylabel("x$_2$")
        plt.savefig('image.pdf',bbox_inches='tight')
        plt.show()
        
        
        
        
        
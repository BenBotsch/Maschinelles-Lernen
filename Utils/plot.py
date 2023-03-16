#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 13:38:58 2022

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt


def plot(x:list,
         y:list,
         xlabel:str,
         ylabel:str,
         xlim:tuple=None,
         ylim:tuple=None,
         legend:list=None,
         legend_loc:str='upper left',
         figsize_x:float=9,
         figsize_y:float=5):
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=((figsize_x*cm, figsize_y*cm)),dpi=600)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.grid(True)
    if(len(x)==0):
        for i in range(len(y)):
            plt.plot(y[i])
    else:
        for i in range(len(y)):
            plt.plot(x[i],y[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(xlim!=None):
        plt.xlim(xlim)
    if(ylim!=None):
        plt.ylim(ylim)
    if(legend!=None):
        plt.legend(legend,loc=legend_loc)
    plt.savefig('image.pdf',bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    x=np.array([0,1,2])
    y=np.array([0,1,2])
    plot([x], [y], "x", "y")
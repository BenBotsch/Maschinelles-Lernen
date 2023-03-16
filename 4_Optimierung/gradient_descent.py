# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:42:14 2022

@author: botsch
"""

import sys
sys.path.append("../Utils")
from plot import plot
import numpy as np
import matplotlib.pyplot as plt
eps = np.finfo(float).eps

class GradientDescent:
    """
    A class that implements the gradient descent algorithm.
    
    Attributes:
    -----------
        param_hist : list
            A list containing the parameter values 
            at each step of the algorithm.
    """
    def __init__(self):
        self.param_hist=[]
    def fit(self, X, y, gradient, start, alpha=0.1, n_iter=500):
        """
        Fits the model to the data using gradient descent.

        Parameters:
        -----------
            X : array-like 
                The input data
            y : array-like 
                The target values
            gradient : function
                The gradient of the loss function 
                to be minimized
            start : array-like
                The initial parameter values
            alpha : float
                The learning rate
            n_iter : int
                The maximum number of iterations
        """
        self.param_hist.append(start)
        self.param = start
        for _ in range(n_iter):
            for i in range(len(y)):
                param = self.param - alpha * np.array(gradient(X[i],y[i],self.param))
                self.param_hist.append(param)
                self.param = param
                
class Momentum:
    """
    A class implementing stochastic gradient descent 
    with momentum.
    """
    def __init__(self):
        self.param_hist=[]
    def fit(self, X, y, gradient, start, alpha=0.1, beta=0.8, n_iter=500):
        """
        Train the model using stochastic gradient descent 
        with momentum.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data for the model.
        y : array-like of shape (n_samples,)
            The target labels for the input data.
        gradient : function
            A function that computes the gradient of 
            the loss function with respect to the model parameters.
        start : array-like of shape (n_features,)
            The initial parameter values for the model.
        alpha : float, optional
            The learning rate for the model.
        beta : float, optional 
            The momentum coefficient for the model.
        n_iter : int, optional
            The number of iterations to run the algorithm for.

        Returns:
        --------
        None

        """
        self.param_hist.append(start)
        self.param = start
        v=0
        for _ in range(n_iter):
            for i in range(len(y)):
                v = beta * v - alpha * np.array(gradient(X[i],y[i],self.param))
                self.param = self.param + v
                self.param_hist.append(self.param)
                #self.param = param
                
class AdaGrad:
    """
    A class implementing AdaGrad algorithm for stochastic 
    gradient descent.
    """
    def __init__(self):
        self.param_hist=[]
    def fit(self, X, y, gradient, start, alpha=0.1, n_iter=500):
        """
        Train the model using AdaGrad algorithm for stochastic gradient descent.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data for the model.
        y : array-like of shape (n_samples,)
            The target labels for the input data.
        gradient : function
            A function that computes the gradient of the loss 
            function with respect to the model parameters.
        start : array-like of shape (n_features,)
            The initial parameter values for the model.
        alpha : float, optional
            The learning rate for the model.
        n_iter : int, optional
            The number of iterations to run the algorithm for.

        Returns:
        --------
        None

        """
        self.param_hist.append(start)
        self.param = start
        G=0
        for _ in range(n_iter):
            for i in range(len(y)):
                g = np.array(gradient(X[i],y[i],self.param)).reshape(-1,1)
                G += g**2
                param = self.param - alpha / (np.sqrt(G+eps)) * g
                self.param_hist.append(param)
                self.param = param
                
class AdamOptim:
    """
    A class for the Adam optimization algorithm.
    """
    def __init__(self, beta1=0.9, beta2=0.999):
        """
        Initializes the AdamOptim object.

        Parameters:
        -----------
        beta1 : float, optional
            The exponential decay rate for the first moment 
            estimate (default is 0.9).
        beta2 : float, optional
            The exponential decay rate for the second moment 
            estimate (default is 0.999).
        
        """
        self.param_hist=[]
        self.m_dg, self.v_dg = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
    def fit(self, X, y, gradient, start, t=1, alpha=0.1, n_iter=500):
        """
        Fits the model to the data using the Adam 
        optimization algorithm.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        gradient : callable
            A function that computes the gradient of the loss 
            function with respect to the parameters.
        start : array-like of shape (n_features,)
            The initial parameter values.
        t : int, optional
            The current iteration number.
        alpha : float, optional
            The learning rate.
        n_iter : int, optional
            The number of iterations.

        """
        self.param = start
        for _ in range(n_iter):
            for i in range(len(y)):
                t+=1
                g = np.array(gradient(X[i],y[i],self.param)).reshape(-1,1)
                self.m_dg = self.beta1*self.m_dg + (1-self.beta1)*g
                self.v_dg = self.beta2*self.v_dg + (1-self.beta2)*(g**2)
                m_dg_corr = self.m_dg/(1-self.beta1**t)
                v_dg_corr = self.v_dg/(1-self.beta2**t)
                param = self.param - alpha * (m_dg_corr/(np.sqrt(v_dg_corr)+eps))
                self.param_hist.append(param)
                self.param = param
    
def compute_cost(X, y, theta=np.array([[0],[0]])):
    m = len(y)
    theta=theta.reshape(2,1)
    yp = np.dot(X,theta)
    error_term = sum((yp - y)**2)
    loss = error_term/(2*m)
    return loss

def gradient(x,y,param):
    return [2*((param[0]+param[1]*x)-y),2*x*((param[0]+param[1]*x)-y)]
    
if __name__ == "__main__":
    
    x = np.array([0.39,0.10,0.30,0.35,0.85]).reshape(-1,1)
    y = np.array([9.83,2.27,5.10,6.32,15.50]).reshape(-1,1)
    gd = GradientDescent()
    gd.fit(
        X=x,
        y=y,
        gradient=gradient, 
        start=np.array([-10.0,-10.0]).reshape(-1,1), 
        alpha=0.01)
    param_hist_gd = np.array(gd.param_hist)
    gd = Momentum()
    gd.fit(
        X=x,
        y=y,
        gradient=gradient, 
        start=np.array([-10.0,-10.0]).reshape(-1,1), 
        alpha=0.01)
    param_hist_mo = np.array(gd.param_hist)
    gd = AdaGrad()
    gd.fit(
        X=x,
        y=y,
        gradient=gradient, 
        start=np.array([-10.0,-10.0]).reshape(-1,1), 
        alpha=0.9)
    param_hist_ag = np.array(gd.param_hist)
    gd = AdamOptim()
    gd.fit(
        X=x,
        y=y,
        gradient=gradient, 
        start=np.array([-10.0,-10.0]).reshape(-1,1), 
        alpha=0.1)
    param_hist_ao = np.array(gd.param_hist)
    
    
    param0 = np.linspace(-20,50,50)
    param1 = np.linspace(-10,25,50)
    J_vals = np.zeros((len(param0), len(param1)))
    
    # compute cost for each combination of theta
    X = np.concatenate((np.ones((len(x),1)), x.reshape(-1,1)), axis=1)
    y = y.reshape(-1,1)
    c1=0; c2=0
    for i in param0:
        for j in param1:
            t = np.array([i, j])
            J_vals[c1][c2] = compute_cost(X, y, t.transpose()).tolist()[0]
            c2=c2+1
        c1=c1+1
        c2=0 # reinitialize to 0
        
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=((9*cm, 5*cm)),dpi=600)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.contour(param0, param1, J_vals.T, levels = np.logspace(-4,3,20),cmap="plasma") 
    plt.plot(param_hist_gd[:,0], param_hist_gd[:,1], 'r.-');
    plt.plot(param_hist_mo[:,0], param_hist_mo[:,1], 'b.-');
    plt.plot(param_hist_ag[:,0], param_hist_ag[:,1], 'g.-');
    plt.plot(param_hist_ao[:,0], param_hist_ao[:,1], 'y.-');

    plt.grid(True)
    plt.xlabel("a")
    plt.ylabel("b")
    plt.legend(["Gradient Descent", "Momentum", "AdaGrad", "Adam"],loc='upper right')
    plt.savefig('image.pdf',bbox_inches='tight')
    plt.show()
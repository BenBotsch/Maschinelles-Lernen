#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:31:26 2022

@author: ben
"""

import sys
sys.path.append("../Utils")
import numpy as np
from numpy import array, matrix, zeros, diag, diagflat, dot, matmul, ones
from numpy.linalg import norm, inv, solve
from numpy.random import randn
from metrics import MeanSquaredError
from dataset import arctan_fun, two_class_example, three_class_example
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt



class LinearRegression:
    """
    A class for linear regression.
    """
    def __init__(self, metric):
        """
        Initializes the LinearRegression object.

        Parameters:
        -----------
        metric : callable
            The error metric to be used for evaluation.
        """
        self.metric = metric
    def fit(self,X,y):
        """
        Fits the model to the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.
        """
        self.beta = solve(X.T.dot(X),X.T.dot(y))
        yp = X.dot(self.beta)
        error = self.metric(y,yp)
        print(error)
    def predict(self,X):
        """
        Predicts the output values for new input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        array-like of shape (n_samples,)
            The predicted output values.
        """
        return X.dot(self.beta)

class IterativeLinearRegression(LinearRegression):
    """
    Implements linear regression using iterative 
    gradient descent.

    Inherits from LinearRegression class.

    Parameters:
    -----------
    metric : callable
        The loss function to use for evaluating 
        model performance.

    Attributes:
    -----------
    beta : numpy.ndarray
        The learned regression coefficients.

    Methods:
    --------
    fit(X, y, iterations=500, eta=0.0001):
        Fits the linear regression model to the given 
        training data using iterative gradient descent.

    predict(X):
        Predicts the output for the given input data 
        using the learned regression coefficients.
    """
    def fit(self,X,y,iterations=500,eta=0.0001):
        self.beta = randn(X.shape[1],1)
        for i in range(iterations):
            gradients = 2*X.T.dot(X.dot(self.beta)-y)
            self.beta = self.beta-eta*gradients
        yp = X.dot(self.beta)
        error = self.metric(y,yp)
        print(error)
        
class LogisticRegression(LinearRegression):
    """
    Logistic regression class for binary 
    classification problems.
    """
    def fit(self,X,y,iterations=500,eta=0.0001):
        """
        Fits the logistic regression model to the 
        training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples, 1)
            Target values.
        iterations : int, optional
            Number of iterations to perform 
            gradient descent.
        eta : float, optional
            Learning rate for gradient descent.
        """
        self.beta = randn(X.shape[1],1)
        for i in range(iterations):
            sigmoid = 1 / (1 + np.exp(-(X.dot(self.beta))))
            gradients = X.T.dot(sigmoid-y)
            self.beta = self.beta-eta*gradients
        yp = self.predict(X)
        error = self.metric(y,yp)
        print(error)
    def predict(self,X):
        """
        Predicts the binary class labels for new data 
        using the fitted model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns:
        --------
        y : array-like, shape (n_samples, 1)
            Predicted binary class labels.
        """
        yp = 1 / (1 + np.exp(-(X.dot(self.beta))))
        yp[yp<0.5] = 0
        yp[yp>0] = 1
        return yp
        
class SoftmaxRegression(LinearRegression):
    """
    A class for performing Softmax Regression.
    """
    def fit(self,X,y,iterations=500,eta=0.0001):
        """
        Fits the Softmax Regression model to the 
        given data.

        Parameters:
        -----------
        X : numpy array
            The feature matrix.

        y : numpy array
            The target vector.

        iterations : int, optional
            The number of iterations for which to 
            run the training.

        eta : float, optional
            The learning rate for the model.

        Returns:
        --------
        None
        """
        self.classes = np.max(y)+1
        self.beta = randn(X.shape[1],int(self.classes))
        self.ohe = OneHotEncoder().fit(y)
        y = self.ohe.transform(y).toarray()
        for i in range(iterations):
            score = np.exp(X.dot(self.beta))
            softmax = score / (np.sum(score,axis=1)).reshape(-1,1)
            gradients = X.T.dot(softmax-y)
            self.beta = self.beta-eta*gradients
        yp = self.predict(X).reshape(-1,1)
        error = self.metric(self.ohe.inverse_transform(y),yp)
        print(error)
        
    def predict(self,X):
        """
        Predicts the target values for the given 
        feature matrix.

        Parameters:
        -----------
        X : numpy array
            The feature matrix.

        Returns:
        --------
        numpy array
            The predicted target values.
        """
        score = np.exp(X.dot(self.beta))
        softmax = score / (np.sum(score,axis=1)).reshape(-1,1)
        yp = np.argmax(softmax,axis=1)
        return yp
        



if __name__ == "__main__":
    run_example=2
    if(run_example==0):
        x,y = arctan_fun()
        poly_feature = PolynomialFeatures(degree=4,include_bias=False)
        scaler = StandardScaler()
        x_poly = poly_feature.fit_transform(x)
        x_scaler = scaler.fit_transform(x_poly)
        x_scaler = np.c_[ones((x_scaler.shape[0],1)),x_scaler]
        mse = MeanSquaredError
        #model = LinearRegression(metric=mse)
        model = IterativeLinearRegression(metric=mse)
        model.fit(x_scaler,y,iterations=1000, eta=0.001)
        yp = model.predict(x_scaler)
    
        cm = 1/2.54  # centimeters in inches
        plt.figure(figsize=((9*cm, 5*cm)),dpi=600)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.grid(True)
        plt.plot(x,y,'black')
        plt.plot(x,yp, 'orange')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig('image.pdf',bbox_inches='tight')
        plt.show()
    elif(run_example==1):
        x,y = two_class_example()
        poly_feature = PolynomialFeatures(degree=3,include_bias=False)
        scaler = StandardScaler()
        mse = MeanSquaredError
        x_poly = poly_feature.fit_transform(x)
        x_scaler = scaler.fit_transform(x_poly)
        x_scaler = np.c_[ones((x_scaler.shape[0],1)),x_scaler]
        model = LogisticRegression(metric=mse)
        model.fit(x_scaler,y,iterations=1000, eta=0.001)
        x_new=np.zeros((50*50,2))
        index=0
        for i in np.linspace(0,3,50):
            for j in np.linspace(0,4.5,40):
                x_new[index,:]=[i,j]
                index+=1
        x_poly = poly_feature.fit_transform(x_new)
        x_scaler = scaler.fit_transform(x_poly)
        x_scaler = np.c_[ones((x_scaler.shape[0],1)),x_scaler]
        yp = model.predict(x_scaler)
        
        cm = 1/2.54  # centimeters in inches
        plt.figure(figsize=((4.5*cm, 4*cm)),dpi=600)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.grid(True)
        for i in range(len(yp)):
            if(yp[i]==0):
                plt.plot(x_new[i,0],x_new[i,1],'b.')
            else:
                plt.plot(x_new[i,0],x_new[i,1],'g.')
        plt.xlim((0,3))
        plt.ylim((0,4.5))
        plt.xlabel("x$_1$")
        plt.ylabel("x$_2$")
        plt.savefig('image.pdf',bbox_inches='tight')
        plt.show()
        
    elif(run_example==2):
        x,y = three_class_example()
        poly_feature = PolynomialFeatures(degree=5,include_bias=False)
        scaler = StandardScaler()
        mse = MeanSquaredError
        x_poly = poly_feature.fit_transform(x)
        x_scaler = scaler.fit_transform(x_poly)
        x_scaler = np.c_[ones((x_scaler.shape[0],1)),x_scaler]
        model = SoftmaxRegression(metric=mse)
        model.fit(x_scaler,y,iterations=1000, eta=0.001)
        x_new=np.zeros((50*50,2))
        index=0
        for i in np.linspace(0,4,50):
            for j in np.linspace(0,5,40):
                x_new[index,:]=[i,j]
                index+=1
        x_poly = poly_feature.fit_transform(x_new)
        x_scaler = scaler.fit_transform(x_poly)
        x_scaler = np.c_[ones((x_scaler.shape[0],1)),x_scaler]
        yp = model.predict(x_scaler)
        
        cm = 1/2.54  # centimeters in inches
        plt.figure(figsize=((4.5*cm, 4*cm)),dpi=600)
        plt.rcParams["font.family"] = "Times New Roman"
        plt.grid(True)
        for i in range(len(yp)):
            if(yp[i]==0):
                plt.plot(x_new[i,0],x_new[i,1],'b.')
            elif(yp[i]==1):
                plt.plot(x_new[i,0],x_new[i,1],'g.')
            else:
                plt.plot(x_new[i,0],x_new[i,1],'c.')
        plt.xlim((0,4))
        plt.ylim((0,5))
        plt.xlabel("x$_1$")
        plt.ylabel("x$_2$")
        plt.savefig('image.pdf',bbox_inches='tight')
        plt.show()
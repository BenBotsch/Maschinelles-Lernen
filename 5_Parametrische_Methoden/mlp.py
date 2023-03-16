#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:46:07 2023

@author: ben
"""

import sys
sys.path.append("../Utils")
from dataset import two_class_example
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class MLP:
    """
    Multi-Layer Perceptron (MLP) class for classification tasks.
    """
    def __init__(self, n_input_variables, n_hidden_nodes, n_classes):
        """
        Initializes the MLP with random weights.
        
        Parameters:
        -----------
            n_input_variables : int
                number of input variables
            n_hidden_nodes : int
                number of nodes in the 
                hidden layer
            n_classes : int
                number of output classes
        """
        w1_rows = n_input_variables + 1
        self.w1 = np.random.randn(w1_rows, n_hidden_nodes) * np.sqrt(1 / w1_rows)
        w2_rows = n_hidden_nodes + 1
        self.w2 = np.random.randn(w2_rows, n_classes) * np.sqrt(1 / w2_rows)

    def sigmoid(self,z):
        """
        Computes the sigmoid function for the 
        given input z.
        Parameters:
        -----------
            z : numpy.ndarray
                The input values to the 
                sigmoid function
        Returns:
        --------
            numpy.ndarray :  
                The output values of the 
                sigmoid function
        """
        return 1 / (1 + np.exp(-z))
    
    
    def softmax(self,logits):
        """
        Computes the softmax function for the given logits.
        Parameters:
        -----------
            logits : numpy.ndarray
                The input logits.
        Returns:
        --------
            numpy.ndarray :  
                The output probabilities 
                of the softmax function.
        """
        exponentials = np.exp(logits)
        return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)
    
    
    def sigmoid_gradient(self,sigmoid):
        """
        Computes the gradient of the sigmoid function 
        for the given sigmoid value.
        Parameters:
        -----------
            sigmoid : numpy.ndarray
                The output values of 
                the sigmoid function.
        Returns:
        --------
            numpy.ndarray :  
                The gradient values of the 
                sigmoid function.
        """
        return np.multiply(sigmoid, (1 - sigmoid))
    
    
    def loss(self,Y, y_hat):
        """
        Calculates the cross-entropy loss between 
        the predicted and actual output values.
        Parameters:
        -----------
            Y : numpy.ndarray
                A numpy array of shape 
                (num_samples, num_classes) 
                containing the actual output 
                values for the given input 
                samples.
            y_hat : numpy.ndarray
                A numpy array of shape 
                (num_samples, num_classes) 
                containing the predicted 
                output values for the 
                given input samples.
        Returns:
        --------
            float :  
                The calculated cross-entropy loss between 
                the predicted and actual output values.
        """
        return -np.sum(Y * np.log(y_hat)) / Y.shape[0]
    
    def prepend_bias(self, X):
        """
        Adds a column of 1s at the beginning of the 
        input array X.
        Parameters:
        -----------
            X : numpy.ndarray
                The input array to which 
                the bias term is to be 
                prepended.
        Returns:
        --------
            numpy.ndarray :  
                The input array with a column 
                of 1s prepended.
        """
        return np.insert(X, 0, 1, axis=1)
    
    def forward(self, X):
        """
        The method takes a numpy array X as input and 
        returns the predicted output and hidden layer 
        values for the input.
        Parameters:
        -----------
            X : numpy.ndarray
                The input array of shape 
                (num_samples, num_features).
        Returns:
        --------
            tuple : (numpy.ndarray, numpy.ndarray)
                A tuple containing 
                two numpy arrays
        """
        h = self.sigmoid(np.matmul(self.prepend_bias(X), self.w1))
        y_hat = self.softmax(np.matmul(self.prepend_bias(h), self.w2))
        return (y_hat, h)
    
    def back(self,X, Y, y_hat, h):
        """
        Calculates the gradients of the weights w1 
        and w2 with respect to the loss function
        using backpropagation algorithm.
        Parameters:
        -----------
            X : numpy.ndarray 
                The input data of shape 
                (num_samples, num_features).
            Y : numpy.ndarray
                The one-hot encoded target 
                labels of shape (num_samples, 
                num_classes).
            y_hat : numpy.ndarray 
                The predicted probabilities 
                of shape (num_samples, 
                num_classes).
            h : numpy.ndarray
                The output of the hidden layer 
                after applying the sigmoid 
                function.
        Returns:
        --------
            Tuple : (numpy.ndarray, numpy.ndarray)
                The gradients of the weights w1 and 
                w2, respectively.
        """
        w2_gradient = np.matmul(self.prepend_bias(h).T, (y_hat - Y)) / X.shape[0]
        w1_gradient = np.matmul(self.prepend_bias(X).T, np.matmul(y_hat - Y, self.w2[1:].T)
                                * self.sigmoid_gradient(h)) / X.shape[0]
        return (w1_gradient, w2_gradient)
    
    def classify(self,X):
        """
        The classify method takes a set of input features 
        and returns the predicted class labels based on 
        the learned weights of the neural network.
        Parameters:
        -----------
            X : numpy.ndarray
                A matrix of shape (n_samples, 
                n_features) containing the input 
                features for which class labels 
                are to be predicted.
        Returns:
        --------
            numpy.ndarray : 
                A matrix of shape (n_samples, 1) 
                containing the predicted class 
                labels for the input features.
        """
        y_hat, _ = self.forward(X)
        labels = np.argmax(y_hat, axis=1)
        return labels.reshape(-1, 1)
    
    def report(self, iteration, X_train, Y_train, X_test, Y_test):
        """
        Computes and prints the training loss and test 
        accuracy for the current iteration.
        Parameters:
        -----------
            iteration (int): 
                The current iteration number.
            X_train : numpy.ndarray
                The training data matrix.
            Y_train : numpy.ndarray
                The training data labels.
            X_test : numpy.ndarray
                The test data matrix.
            Y_test : numpy.ndarray
                The test data labels.
        Returns:
        --------
            None
        """
        y_hat, _ = self.forward(X_train)
        training_loss = self.loss(Y_train, y_hat)
        classifications = self.classify(X_test)
        accuracy = np.average(classifications == Y_test) * 100.0
        print("Iteration: %5d, Loss: %.8f, Accuracy: %.2f%%" %
              (iteration, training_loss, accuracy))
        
    def train(self, X_train, Y_train, X_test, Y_test, iterations=1000, lr=0.01):
        """
        Train the neural network model using the 
        given input data and hyperparameters.
        Parameters:
        -----------
            X_train : numpy.ndarray
                Input features for 
                training the model.
            Y_train : numpy.ndarray
                Target output values 
                for the training data.
            X_test : numpy.ndarray
                Input features for 
                testing the model.
            Y_test : numpy.ndarray
                Target output values 
                for the testing data.
            iterations : int
                Number of training iterations 
                to perform.
            lr : float
                Learning rate to use for training.
        Returns:
        --------
            None
        """
        for iteration in range(iterations):
            y_hat, h = self.forward(X_train)
            w1_gradient, w2_gradient = self.back(X_train, Y_train, y_hat, h)
            self.w1 = self.w1 - (w1_gradient * lr)
            self.w2 = self.w2 - (w2_gradient * lr)
            self.report(iteration, X_train, Y_train, X_test, Y_test)
    





if __name__ == "__main__":

    # Erstelle Beispiel-Daten für Regressionsproblem
    X, y = two_class_example()
    X, y = shuffle(X, y, random_state=0)
    y_ohe = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    
    # Erstelle MLP-Modell mit 2 versteckten Schichten (jeweils 10 Neuronen)
    mlp = MLP(2,30,2)
    
    # Trainiere das Modell für 1000 Epochen mit Lernrate 0.01
    mlp.train(X[0:-20,:], y_ohe[0:-20,:], X[-20:,:], y[-20:], iterations=9000)
    

    x_new=np.zeros((50*50,2))
    index=0
    for i in np.linspace(0,3,50):
        for j in np.linspace(0,4.5,40):
            x_new[index,:]=[i,j]
            index+=1

    yp = mlp.classify(x_new)
    
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
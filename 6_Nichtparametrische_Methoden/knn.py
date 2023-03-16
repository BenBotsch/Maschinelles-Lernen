# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:10:34 2023

@author: botsch
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter




class KNN:
    """
    This class implements a k-Nearest Neighbors Classifier, 
    which can be used for classification tasks.
    """
    def __init__(self, k=5):
        """
        Initializes the KNN model with a default k 
        value of 5.
        Parameters:
        -----------
            k : int
                The number of neighbors to 
                consider for classification.
        """
        self.k = k

    def fit(self, X, y):
        """
        Trains the KNN model on the input data X 
        and its corresponding labels y.
        Parameters:
        -----------
            X : numpy.ndarray
                The input data to train 
                the KNN model on.
            y : numpy.ndarray
                The corresponding labels 
                for the input data X.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the labels for the input data X using 
        the trained KNN model.
        Parameters:
        -----------
            X : numpy.ndarray
                The input data to predict 
                the labels for.
        Returns:
        --------
            numpy.ndarray : 
                An array of predicted labels 
                for the input data X.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Helper function for the predict method that 
        calculates the predicted label for a single 
        data point.
        Parameters:
        -----------
            x : numpy.ndarray
                A single data point to 
                predict the label for.
        Returns:
        --------
            int : 
                The predicted label for the input 
                data point x.
        """
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    
    def euclidean_distance(self, x1, x2):
        """
        Calculates the Euclidean distance between two 
        data points.
        Parameters:
        -----------
            x1 : numpy.ndarray
                The first data point.
            x2 : numpy.ndarray
                The second data point.
        Returns:
        --------
            float :
                The Euclidean distance between the 
                two data points.
        """
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def accuracy(self, y_true, y_pred):
        """
        Calculates the accuracy of the KNN model on 
        the input data.
        Parameters:
        -----------
            y_true : numpy.ndarray
                The true labels for 
                the input data.
            y_pred : numpy.ndarray
                The predicted labels 
                for the input data.
        Returns:
        --------
            float :  
                The accuracy of the KNN model on the 
                input data.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    

if __name__ == "__main__":
    X = np.array([[1,1],[1.5,2],[2,0], [3.2,1.5],[4,2],[2.7,3]])
    y = np.array([-1,-1,-1, 1,1,1])
    plt.scatter(X[0:3,0], X[0:3,1], marker='o')
    plt.scatter(X[3:,0], X[3:,1], marker='x')
    plt.show()
    
    knn = KNN()
    knn.fit(X, y)
    yp = knn.predict(X)
    print("KNN classification accuracy", knn.accuracy(y, yp))
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:26:38 2023

@author: botsch
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class NaiveBayes:
    """
    The NaiveBayes class implements a Naive Bayes algorithm 
    for classification.
    """
    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to a training set.

        Parameters:
        -----------
        X (numpy.ndarray):  The training data, where each 
                            row represents a sample and each 
                            column represents a feature.
        y (numpy.ndarray): The target values for the training data.

        Returns:
        --------
        None
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predict the target values for a test set.

        Parameters:
        -----------
        X (numpy.ndarray):  The test data, where each row 
                            represents a sample and each column 
                            represents a feature.

        Returns:
        --------
        numpy.ndarray:  The predicted target values for 
                        the test data.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the Naive Bayes classifier.

        Parameters:
        -----------
        y_true (numpy.ndarray): The true target values.
        y_pred (numpy.ndarray): The predicted target values. 

        Returns:
        --------
        float: The accuracy of the Naive Bayes classifier.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def _predict(self, x):
        """
        Predict the target value for a single sample.

        Parameters:
        -----------
        x (numpy.ndarray): A single sample.

        Returns:
        --------
        float: The predicted target value for the sample.
        """
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function for a 
        single feature of a single sample.

        Parameters:
        -----------
        class_idx (int): The index of the class to 
                         calculate the PDF for.
        x (numpy.ndarray): A single sample.

        Returns:
        --------
        numpy.ndarray:  The calculated PDF for the 
                        specified feature and class.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    
    
if __name__ == "__main__":
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123)
    plt.scatter(X[:,0], X[:,1], marker='.')
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)
    
    nb = NaiveBayes()
    nb.fit(X, y)
    predictions = nb.predict(X_test)
    
    print("Naive Bayes classification accuracy", nb.accuracy(y_test, predictions))
    
    
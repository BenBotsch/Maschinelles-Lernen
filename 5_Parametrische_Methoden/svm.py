# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:53:22 2023

@author: botsch
"""

import numpy as np
import matplotlib.pyplot as plt

class SVM:
    """
    A class for performing Support Vector 
    Machine (SVM) classification.
    """
    def __init__(self, learning_rate=0.01, alpha=0.01, n_iters=5000):
        """
        Initializes the SVM object.

        Parameters:
        -----------
        learning_rate : float, optional
            The learning rate for the model.

        alpha : float, optional
            The regularization parameter for the model. 

        n_iters : int, optional
            The number of iterations for which to 
            run the training.

        Returns:
        --------
        None
        """
        self.lr = learning_rate
        self.alpha = alpha
        self.n_iters = n_iters
        self.w = None
        self.w0 = None

    def fit(self, X, y):
        """
        Fits the SVM model to the given data.

        Parameters:
        -----------
        X : numpy array
            The feature matrix.

        y : numpy array
            The target vector.

        Returns:
        --------
        None
        """
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.w0 = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.w0) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.alpha * self.w)
                else:
                    self.w -= self.lr * (2 * self.alpha * self.w - np.dot(x_i, y_[idx]))
                    self.w0 -= self.lr * y_[idx]

    def predict(self, X):
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
        approx = np.dot(X, self.w) - self.w0
        return np.sign(approx)
    
    
    
if __name__ == "__main__":
    X = np.array([[1,1],[1.5,2],[2,0], [3.2,1.5],[4,2],[2.7,3]])
    y = np.array([-1,-1,-1, 1,1,1])
    
    svm = SVM()
    svm.fit(X, y)
    
    yp = svm.predict(X)
    
    x1=1
    x2=3.5
    y1 = -(x1*svm.w[0]-svm.w0)/svm.w[1]
    y2 = -(x2*svm.w[0]-svm.w0)/svm.w[1]
    
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=((8*cm, 5*cm)),dpi=600)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.grid(True)
    plt.plot(X[0:3,0], X[0:3,1], 'bo')
    plt.plot(X[3:,0], X[3:,1], 'mx')
    plt.plot([x1,x2],[y1,y2],'--')
    plt.xlabel('x$_1$')
    plt.ylabel('x$_2$')
    plt.savefig('image.pdf',bbox_inches='tight')
    plt.show()

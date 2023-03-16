#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:21:02 2023

@author: ben
"""


import numpy as np



class LDA:
    def __init__(self):
        self.w = None
        self.mean = None
        self.classes = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.cov = np.cov(X.T)
        self.cov_inv = np.linalg.inv(self.cov + 1e-8 * np.eye(self.cov.shape[0]))
        
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            
        self.w = {}
        for i, c1 in enumerate(self.classes):
            self.w[c1] = []
            for c2 in self.classes[i+1:]:
                self.w[c1].append(np.dot(self.cov_inv, self.mean[c1] - self.mean[c2]))
        
    def predict(self, X):
        pred = []
        for x in X:
            scores = {c: np.dot(x, w) for c, ws in self.w.items() for w in ws}
            pred.append(max(scores, key=scores.get))
        return np.array(pred)


if __name__ == "__main__":

    
    X = np.array([[1.1, 4.3], [3.5, 2.1], [0.9, 5.1], [1.6, 4.0], [5, 2.6], [6, 2.9]])
    y = np.array([0, 1, 0, 0, 1, 1])
    
    clf = LDA()
    clf.fit(X, y)
    predictions = clf.predict(X)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:31:59 2023

@author: ben
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def plot(class1,class2,class3):
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=((7*cm, 5*cm)),dpi=600)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.grid(True)
    plt.plot(class1[0],class1[1],'b.')
    plt.plot(class2[0],class2[1],'g.')
    plt.plot(class3[0],class3[1],'c.')
    #plt.xlim((0,3))
    #plt.ylim((0,4.5))
    plt.xlabel("x$_1$")
    plt.ylabel("x$_2$")
    plt.savefig('image.pdf',bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", delimiter=",")
    
    
    X = df.copy().to_numpy()
    y = X[:,0]
    X = X[:,1:]
    X_pca = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X_pca)
    
    plot([pca_results[0:57,0],pca_results[0:57,1]], [pca_results[58:128,0],pca_results[58:128,1]], [pca_results[129:,0],pca_results[129:,1]])
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    print("Accuracy score: " + str(accuracy_score(y_test, y_pred)))
    print("\nConfusion matrix: \n" + str(confusion_matrix(y_test, y_pred)))
    print("\nClassification report: \n" + str(classification_report(y_test, y_pred)))
    
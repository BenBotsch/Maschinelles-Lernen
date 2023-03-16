#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:01:01 2023

@author: ben
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist 


class KMeans:
    """
    A class for performing k-means clustering on input data.
    """
    def __init__(self, k: int=2, iterations: int=1000):
        """
        Initializes a new KMeans object.
        Parameters:
        -----------
        k : int
            Number of clusters to create.
        iterations : int
            Number of iterations to perform.
        """
        self.k = k
        self.iterations = iterations

    def train(self, x):
        """
        Performs k-means clustering on input data x.
        Parameters:
        -----------
        x : numpy.ndarray
            Input data to cluster.
        Returns:
        --------
        numpy.ndarray : 
            Array of cluster assignments 
            for each data point.
        """
        idx = np.random.choice(len(x), self.k, replace=False)
        #Randomly choosing Centroids 
        centroids = x[idx, :] #Step 1
         
        #finding the distance between centroids and all the data points
        distances = cdist(x, centroids ,'euclidean') #Step 2
         
        #Centroid with the minimum Distance
        points = np.array([np.argmin(i) for i in distances]) #Step 3
         
        #Repeating the above steps for a defined number of iterations
        for _ in range(self.iterations): 
            centroids = []
            for idx in range(self.k):
                #Updating Centroids by taking mean of Cluster it belongs to
                temp_cent = x[points==idx].mean(axis=0) 
                centroids.append(temp_cent)
     
            centroids = np.vstack(centroids) #Updated Centroids 
             
            distances = cdist(x, centroids ,'euclidean')
            points = np.array([np.argmin(i) for i in distances])
             
        return points
                
                
                
                
                
if __name__ == "__main__":
    


    n_samples = 1500
    random_state = 170
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_varied, y_varied = make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    
    clusterer = KMeans(3)
    label = clusterer.train(X_varied)
    
    #Visualize the results
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(X_varied[label[:]==i,0], X_varied[label[:]==i,1] , label = i)
    plt.legend()
    plt.show()
    
    
    
    
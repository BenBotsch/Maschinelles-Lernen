#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:31:29 2023

@author: ben
"""

import math
import matplotlib.pyplot as plt

def distance(p, q):
    """Compute the Euclidean distance between two points."""
    return math.sqrt(sum([(pi - qi)**2 for pi, qi in zip(p, q)]))


def single_link(ci, cj):
    """Compute the distance between the two closest points of two clusters."""
    return min([distance(vi, vj) for vi in ci for vj in cj])


def complete_link(ci, cj):
    """Compute the distance between the two furthest points of two clusters."""
    return max([distance(vi, vj) for vi in ci for vj in cj])


def average_link(ci, cj):
    """Compute the average distance between all pairs of points from two clusters."""
    distances = [distance(vi, vj) for vi in ci for vj in cj]
    return sum(distances) / len(distances)


def get_distance_measure(M):
    """
    Return the appropriate distance measure function based on the input parameter.

    Parameters
    ----------
    M : int
        Indicates the distance measure to use. 0 = single link, 1 = complete link,
        and anything else = average link.

    Returns
    -------
    function
        The selected distance measure function.
    """
    if M == 0:
        return single_link
    elif M == 1:
        return complete_link
    else:
        return average_link
    
class AgglomerativeHierarchicalClustering:
    """
    Class for performing agglomerative hierarchical clustering on a given dataset.

    Parameters
    ----------
    data : list
        A list of n-dimensional points to cluster.
    K : int
        The desired number of clusters to form.
    M : int
        Indicates the distance measure to use. 0 = single link, 1 = complete link,
        and anything else = average link.
    """
    def __init__(self, data, K, M):
        self.data = data
        self.N = len(data)
        self.K = K
        self.measure = get_distance_measure(M)
        self.clusters = self.init_clusters()

    def init_clusters(self):
        """
        Initialize each data point as a separate cluster.

        Returns:
            A dictionary mapping cluster IDs to lists of points belonging to each cluster.
        """
        return {data_id: [data_point] for data_id, data_point in enumerate(self.data)}

    def find_closest_clusters(self):
        """
        Find the two closest clusters based on the chosen distance measure.

        Returns
        -------
        tuple
            The ids of the two closest clusters as a tuple.
        """
        min_dist = math.inf
        closest_clusters = None

        clusters_ids = list(self.clusters.keys())

        for i, cluster_i in enumerate(clusters_ids[:-1]):
            for j, cluster_j in enumerate(clusters_ids[i+1:]):
                dist = self.measure(self.clusters[cluster_i], self.clusters[cluster_j])
                if dist < min_dist:
                    min_dist, closest_clusters = dist, (cluster_i, cluster_j)
        return closest_clusters

    def merge_and_form_new_clusters(self, ci_id, cj_id):
        """
        Merge two clusters and create a new dictionary of clusters.

        Parameters
        ----------
        ci_id : int
            The id of the first cluster to merge.
        cj_id : int
            The id of the second cluster to merge.

        Returns
        -------
        dict
            A dictionary of clusters, where each cluster is a list of data points.
        """
        new_clusters = {0: self.clusters[ci_id] + self.clusters[cj_id]}

        for cluster_id in self.clusters.keys():
            if (cluster_id == ci_id) | (cluster_id == cj_id):
                continue
            new_clusters[len(new_clusters.keys())] = self.clusters[cluster_id]
        return new_clusters

    def run_algorithm(self):
        """
        Run the agglomerative hierarchical clustering algorithm on the data.
    
        This function repeatedly merges the two closest clusters until the desired number of clusters is reached.
    
        Args:
            None
    
        Returns:
            None
        """
        while len(self.clusters.keys()) > self.K:
            closest_clusters = self.find_closest_clusters()
            self.clusters = self.merge_and_form_new_clusters(*closest_clusters)

    def print(self):
        """
        Print the clusters and their constituent points.
        
        This function prints the ID of each cluster, followed by the points that belong to that cluster.
        
        Args:
            None
        
        Returns:
            None
        """
        for id, points in self.clusters.items():
            print("Cluster: {}".format(id))
            for point in points:
                print("    {}".format(point))
                
                
if __name__ == "__main__": 
     x = [[113.5469, 31.2294],
          [-79.0333, 9.5167],
          [-122.1018, 47.9934],
          [147.1510, -41.4384],
          [7.0186, 8.8835],
          [82.6712 ,53.5044],
          [29.9419, -0.8411],
          [3.6912, 36.1477],
          [-89.2853, 32.4513],
          [-73.9976, 40.8482],
          [7.9337, 11.9160],
          [-2.3333, 53.3500],
          [1.3519, 43.4798],
          [153.1532, -27.4919]]
     agg_hierarchical_clustering = AgglomerativeHierarchicalClustering(x, 3, 0)
     agg_hierarchical_clustering.run_algorithm()
     agg_hierarchical_clustering.print()
     
     for id, points in agg_hierarchical_clustering.clusters.items():
         for point in points:
             if(id==0):
                 plt.plot(point[0],point[1],'ro')
             elif(id==1):
                 plt.plot(point[0],point[1],'bo')
             elif(id==2):
                 plt.plot(point[0],point[1],'go')
     plt.show()
#!/usr/bin/env python3
"""Summary
"""
import sklearn.cluster


def kmeans(X, k):
    """

    Args:
      X (numpy.ndarray): shape (n, d) containing the dataset
      k (pos. integer): the number of clusters

    Returns:
        C (numpy.ndarray): shape (k, d) containing the centroid means
            for each cluster
        clss (numpy.ndarray): shape (n,) containing the index of the cluster
            in C that each data point belongs to
    """
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_
    return C, clss

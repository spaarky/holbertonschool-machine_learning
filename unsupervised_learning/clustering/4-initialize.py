#!/usr/bin/env python3
"""Summary
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        k (pos. integer): shape (k, d) containing the number of clusters

    Returns:
        pi (numpy.ndarray) of shape (k,) containing the priors for each
            cluster, initialized evenly
        m (numpy.ndarray) of shape (k, d) containing the centroid means
            for each cluster, initialized with K-means
        S (numpy.ndarray) of shape (k, d, d) containing the covariance
            matrices for each cluster, initialized as identity matrices
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    n, d = X.shape
    centroids, clss = kmeans(X, k)
    pi = np.ones(k) / k
    m = centroids
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))

    return pi, m, S

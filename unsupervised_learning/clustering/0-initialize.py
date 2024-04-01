#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset that
            will be used for K-means clustering
                - n is the number of data points
                - d is the number of dimensions for each data point
        k (pos. integer): number of clusters

    Returns:
        numpy.ndarray: shape (k, d) containing the initialized centroids for
            each cluster, or None on failure
    """

    if type(k) is not int or k <= 0:
        return None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    _, d = X.shape
    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                             size=(k, d))

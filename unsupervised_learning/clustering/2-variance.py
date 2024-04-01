#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        C (numpy.ndarray): shape (k, d) containing the centroid means for
            each cluster

    Returns:
        var (float): containing the total variance, or None on failure
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    k, d = C.shape
    if type(k) is not int or k <= 0:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    D = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=2))
    cluster = np.min(D, axis=0)

    return np.sum(np.square(cluster))

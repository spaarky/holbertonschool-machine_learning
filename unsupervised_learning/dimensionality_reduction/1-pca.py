#!/usr/bin/env python3
"""
Performs PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset

    Args:
        X (numpy.ndarray): shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
        ndim (integer): new dimensionality of the transformed X

    Returns:
        numpy.ndarray: shape (n, ndim) containing the transformed version of X
    """
    X_mean = X - X.mean(axis=0)
    u, s, vh = np.linalg.svd(X_mean)

    W = vh.T
    Wr = W[:, 0:ndim]
    # @ = matrix multiplication operator
    T = X_mean @ Wr

    return T

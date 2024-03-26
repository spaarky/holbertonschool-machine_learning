#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset

    Args:
        X (numpy.ndarray): shape (n, d) where:
            - n is the number of data points
            - d is the number of dimensions in each point
        var (float, optional): fraction of the variance that the PCA
            transformation should maintain. Defaults to 0.95.

    Returns:
        numpy.ndarray: weights matrix
    """
    u, s, vh = np.linalg.svd(X)
    cum = np.cumsum(s)
    thresh = cum[len(cum) - 1] * var
    mask = np.where(thresh > cum)
    var = cum[mask]
    idx = len(var) + 1
    W = vh.T
    Wr = W[:, 0:idx]
    return Wr

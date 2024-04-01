#!/usr/bin/env python3
"""Summary
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        kmin (pos. integer, optional): containing the minimum number of
            clusters to check for (inclusive). Defaults to 1.
        kmax (pos. integer, optional): containing the maximum number of
            clusters to check for (inclusive). Defaults to None.
        iterations (pos. integer, optional): containing the maximu
            number of iterations for K-means. Defaults to 1000.

    Returns:
        results (list) containing the outputs of K-means for each
            cluster size, or None on failure
        d_vars (list) containing the difference in variance from the
            smallest cluster size for each cluster size
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        if k == kmin:
            var_min = variance(X, C)
        var = variance(X, C)
        d_vars.append(var_min - var)

    return results, d_vars

#!/usr/bin/env python3
"""Summary
"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian
        Information Criterion

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        kmin (pos. integer, optional): the minimum number of clusters to
            check for (inclusive)
        kmax (pos. integer, optional): the maximum number of clusters to
            check for (inclusive)
        iterations (int, optional): containing the maximum number of
            iterations for the EM algorithm
        tol (non-neg. float, optional): containing the tolerance for the
            EM algorithm
        verbose (bool, optional): determines if you should print information
            about the algorithm

    Returns:
        best_k (int): best value for k based on its BIC
        best_result (tuple): containing pi, m, S
            pi (numpy.ndarray): shape (k,) containing the cluster priors for
                the best number of clusters
            m (numpy.ndarray): shape (k, d) containing the centroid means for
                the best number of clusters
            S (numpy.ndarray): shape (k, d, d) containing the covariance
                matrices for the best number of clusters
        l (numpy.ndarray): shape (kmax - kmin + 1) containing the log
            likelihood for each cluster size tested
        b (numpy.ndarray): shape (kmax - kmin + 1) containing the BIC value
            for each cluster size tested
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None
    if type(tol) != float or tol <= 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None

    k_best = []
    best_res = []
    logl_val = []
    bic_val = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S,  _, log_l = expectation_maximization(X, k, iterations, tol,
                                                       verbose)
        k_best.append(k)
        best_res.append((pi, m, S))
        logl_val.append(log_l)

        cov_params = k * d * (d + 1) / 2.
        mean_params = k * d
        p = int(cov_params + mean_params + k - 1)

        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)

    bic_val = np.array(bic_val)
    logl_val = np.array(logl_val)
    best_val = np.argmin(bic_val)

    k_best = k_best[best_val]
    best_res = best_res[best_val]

    return k_best, best_res, logl_val, bic_val

#!/usr/bin/env python3
"""Summary
"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """finds the best number of clusters for a GMM using the Bayesian
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

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if type(kmax) is not int or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape

    all_pis = []
    all_ms = []
    all_Ss = []
    all_likelihoods = []
    all_BICs = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(X, k,
                                                               iterations,
                                                               tol, verbose)

        all_pis.append(pi)
        all_ms.append(m)
        all_Ss.append(S)
        all_likelihoods.append(log_likelihood)

        # p is the number of parameters required the model
        p = (k * d * (d + 1) / 2) + (d * k) + (k - 1)

        # b: array containing the BIC value each cluster size tested
        b = p * np.log(n) - 2 * log_likelihood
        all_BICs.append(b)

    all_likelihoods = np.array(all_likelihoods)
    all_BICs = np.array(all_BICs)
    best_k = np.argmin(all_BICs)
    best_result = (all_pis[best_k], all_ms[best_k], all_Ss[best_k])

    return best_k+1, best_result, all_likelihoods, all_BICs

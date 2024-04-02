#!/usr/bin/env python3
"""Summary
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        k (pos. integer): containing the number of clusters
        iterations (pos integer, optional): containing the maximum number of
            iterations for the algorithm. Defaults to 1000.
        tol (non-negative float, optional): containing tolerance of the log
            likelihood, used to determine early stopping i.e. if the
            difference is less than or equal to tol you should stop the
            algorithm. Defaults to 1e-5.
        verbose (bool, optional): determines if you should print information
            about the algorithm. Defaults to False.

    Returns:
        pi (numpy.ndarray) of shape (k,) containing the priors for each
            cluster
        m (numpy.ndarray) of shape (k, d) containing the centroid means for
            each cluster
        S (numpy.ndarray) of shape (k, d, d) containing the covariance
            matrices for each cluster
        g (numpy.ndarray) of shape (k, n) containing the posterior
            probabilities for each data point in each cluster
        log_likelihood (float) is the total log likelihood
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    i = 0
    l_prev = 0
    pi, mean, cov = initialize(X, k)
    g, log_likelihood = expectation(X, pi, mean, cov)

    while i < iterations:
        if (np.abs(l_prev - log_likelihood)) <= tol:
            break
        l_prev = log_likelihood

        if verbose is True and (i % 10 == 0):
            rounded = log_likelihood.round(5)
            print("Log Likelihood after {} iterations: {}".format(i, rounded))

        pi, mean, cov = maximization(X, g)
        g, log_likelihood = expectation(X, pi, mean, cov)
        i += 1

    if verbose is True:
        rounded = log_likelihood.round(5)
        print("Log Likelihood after {} iterations: {}".format(i, rounded))

    return pi, mean, cov, g, log_likelihood

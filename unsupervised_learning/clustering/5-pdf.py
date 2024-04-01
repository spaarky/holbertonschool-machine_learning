#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def pdf(X, m, S):
    """calculates the probability density function of a Gaussian distribution

    Args:
        X (numpy.ndarray): shape (n, d) containing the data points whose PDF
            should be evaluated
        m (numpy.ndarray): shape (d,) containing the mean of the distribution
        S (numpy.ndarray): shape (d, d) containing the covariance of the
            distribution

    Returns:
        P (numpy.ndarray) of shape (n,) containing the PDF values for each
            data point, or None on failure
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    d = S.shape[0]

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    first = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    second = np.dot((X - m), inv)
    third = np.sum(second * (X - m) / -2, axis=1)
    P = first * np.exp(third)
    P = np.maximum(P, 1e-300)

    return P

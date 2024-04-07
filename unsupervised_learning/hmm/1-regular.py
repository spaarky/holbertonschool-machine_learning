#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular markov chain

    Args:
        P (numpy.ndarray): shape (n, n) representing the transition matrix

    Returns:
        numpy.ndarray: shape (1, n) containing the steady state probabilities,
            or None on failure
    """

    if type(P) is not np.ndarray:
        return None
    if len(P.shape) != 2:
        return None
    n, n_t = P.shape
    if n != n_t:
        return None
    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return None

    evals, evecs = np.linalg.eig(P.T)
    stationary = evecs / evecs.sum()
    stationary = stationary.real

    for i in np.dot(stationary.T, P):
        if (i >= 0).all() and np.isclose(i.sum(), 1):
            return i.reshape(1, n)

    return None

#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing

    Args:
        P (numpy.ndarray): shape (n, n) representing the standard
            transition matrix

    Returns:
        bool: True if it is absorbing, False if it is not
    """

    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None
    for i in range(n):
        if np.all(P[i] == np.eye(n)[i]):
            return True
    return False

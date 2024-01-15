#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def normalize(X, m, s):
    """Function that normalizes a matrix

    Args:
        X (numpy.ndarray): matrix to normalize
        m (numpy.ndarray): matrix with the means of all the feature of X
        s (numpy.ndarray): matrix with the std of the feature of X
    """
    return np.divide(np.subtract(X, m), s)

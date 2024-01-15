#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def normalization_constants(X):
    """Function that calculates the normalization constants of a matrix

    Args:
        X (numpy.ndarray): matrix to normalize

    Returns:
        numpy.ndarray: mean and standard deviation of X
    """
    return np.mean(X, axis=0), np.std(X, axis=0)

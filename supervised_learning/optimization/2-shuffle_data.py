#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def shuffle_data(X, Y):
    """Function that shuffles the data points in two matrices the same way

    Args:
        X (numpy.ndarray): matrix to shuffle
        Y (numpy.ndarray): matrix to shuffle

    Returns:
        numpy.ndarray: shuffled matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation, :], Y[permutation, :]

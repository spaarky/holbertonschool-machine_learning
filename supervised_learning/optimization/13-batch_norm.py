#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function that normalizes an unactivated output of a neural
        network using batch normalization

    Args:
        Z (numpy.ndarray): matrices of values to be normalized
        gamma (numpy.ndarray): contains the scales used for batch normalization
        beta (numpy.ndarray): contains the offsets used for batch normalization
        epsilon (float): small number used to avoid division by 0

    Returns:
        numpy.ndarray: normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    normalized = (Z - mean) / np.sqrt(var + epsilon)
    z_n = gamma * normalized + beta
    return z_n

#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Function that updates a variable in place using the Adam
        optimization algorithm

    Args:
        alpha (float): learning rate
        beta1 (float): weight used for the first moment
        beta2 (flaot): weight used for the second moment
        epsilon (float): small number to avoid division by 0
        var (numpy.ndarray): contains the variable to be updated
        grad (numpy.ndarray): contains the gradient of var
        v (float): previous first moment of var
        s (float): previous second moment of var
        t (int): times step used for bias correction

    Returns:
        : updated variable, new first moment, new second moment
    """
    vdw = beta1 * v + (1 - beta1) * grad
    sdw = beta2 * s + (1 - beta2) * grad ** 2
    # Corrected
    vdw_corrected = vdw / (1 - beta1 ** t)
    sdw_corrected = sdw / (1 - beta2 ** t)
    # Updating value
    var = var - alpha * (vdw_corrected / (np.sqrt(sdw_corrected) + epsilon))
    return var, vdw, sdw

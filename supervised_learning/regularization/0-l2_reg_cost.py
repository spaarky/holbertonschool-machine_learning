#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates the cost of a neural network
        with L2 regularization

    Args:
        cost (float): cost of the network without L2 reg
        lambtha (float): reg parameter
        weights (dictionnary): dictionary of the weights and biases
            (numpy.ndarrays) of the neural network
        L (integer): number of layers in the neural network
        m (integer): number of data points used

    Returns:
        (float): cost of the network accounting for L2 regularization
    """
    L2 = 0
    for i in range(L):
        w = weights["W{}".format(i + 1)]
        norm = np.linalg.norm(w)
        L2 += (lambtha / 2 / m * norm)
    return L2 + cost

#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural network
        using gradient descent with L2 regularization

    Args:
        Y (numpy.ndarray): shape (classes, m) that contains the correct
            labels for the data
        weights (dictionnary): dictionary of the weights and biases of
            the neural network
        cache (dictionnary): dictionary of the outputs of each layer of
            the neural network
        alpha (float): learning rate
        lambtha (float): L2 regularization parameter
        L (integer): number of layers of the network
    """
    m = Y.shape[1]
    for i in reversed(range(L)):
        key_w = 'W' + str(i + 1)
        key_b = 'b' + str(i + 1)
        key_cache = 'A' + str(i + 1)
        key_cache_dw = 'A' + str(i)
        A = cache[key_cache]
        A_dw = cache[key_cache_dw]
        if i == L - 1:
            dz = A - Y
            W = weights[key_w]
        else:
            da = 1 - (A * A)
            dz = np.matmul(W.T, dz)
            dz = dz * da
            W = weights[key_w]
        dw = np.matmul(A_dw, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[key_w] = weights[key_w] - alpha * (dw.T + (lambtha / m *
                                                           weights[key_w]))
        weights[key_b] = weights[key_b] - alpha * db

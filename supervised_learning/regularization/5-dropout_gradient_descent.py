#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network with
        Dropout regularization using gradient descent

    Args:
        Y (numpy.ndarray): shape (classes, m) that contains the
            correct labels for the data
        weights (dictionary): dictionary of the weights and biases
            of the neural network
        cache (dictionary): dictionary of the outputs and dropout masks
            of each layer of the neural network
        alpha (float): learning rate
        keep_prob (float): probability that a node will be kept
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
            dz = dz * da * cache["D{}".format(i + 1)]
            dz = dz / keep_prob
            W = weights[key_w]
        dw = np.matmul(A_dw, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[key_w] = weights[key_w] - alpha * dw.T
        weights[key_b] = weights[key_b] - alpha * db

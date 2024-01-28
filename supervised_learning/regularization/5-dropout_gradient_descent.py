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
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        dw = np.matmul(dz, cache['A' + str(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dz = np.matmul(weights['W' + str(i)].T, dz) * (
            1 - cache["A" + str(i - 1)] ** 2)
        if i > 1:
            dz *= cache['D' + str(i - 1)] / keep_prob
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db

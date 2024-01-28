#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that conducts forward propagation using Dropout

    Args:
        X (numpy.ndarray): shape (nx, m) containing the input data for
            the network
        weights (dictionnary): dictionary of the weights and biases of
            the neural network
        L (integer): number of layers in the network
        keep_prob (float): probability that a node will be kept

    Returns:
        (dictionnary): dictionary containing the outputs of each layer
            and the dropout mask used on each layer
    """
    cache = {'A0': X}
    for i in range(L):
        key_w = 'W' + str(i + 1)
        key_b = 'b' + str(i + 1)
        key_cache = 'A' + str(i + 1)
        key_cache_last = 'A' + str(i)
        output_Z = np.matmul(weights[key_w], cache[
            key_cache_last]) + weights[key_b]
        if i == L - 1:
            t = np.exp(output_Z)
            output_A = np.exp(output_Z) / np.sum(t, axis=0,
                                                 keepdims=True)
        else:
            output_A = ((np.exp(output_Z) - np.exp(-output_Z)) / (
                np.exp(output_Z) + np.exp(-output_Z)))
            rand = np.random.rand(output_A.shape[0], output_A.shape[1])
            dropout = (rand < keep_prob).astype(int)
            output_A = np.multiply(output_A, dropout)
            output_A = output_A / keep_prob
            cache['D{}'.format(i + 1)] = dropout
        cache[key_cache] = output_A
    return cache

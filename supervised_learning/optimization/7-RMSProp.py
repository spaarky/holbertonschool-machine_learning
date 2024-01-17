#!/usr/bin/env python3
"""Summary
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Function that updates a variable using the RMSProp
        optimization algorithm

    Args:
        alpha (float): learning rate
        beta2 (float): RMSprop weight
        epsilon (float): small number to avoid division by 0
        var (numpy.ndarray): contain the variable to be updated
        grad (numpy.ndarray): contains the gradient of var
        s (float): previous second moment of var

    Returns:
        : updated variable and the new moment
    """
    new_moment = beta2 * s + (1 - beta2) * grad ** 2
    updated = var - alpha * grad / (new_moment ** (1 / 2) + epsilon)
    return updated, new_moment

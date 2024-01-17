#!/usr/bon/env python3
"""Summary
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Function at updates a variable using the gradient descent with
        momentum optimization algorithm

    Args:
        alpha (float): learning rate
        beta1 (float): momentum weight
        var (numpy.ndarray): contain the variable to be updated
        grad (numpy.ndarray): contain the gradient of var
        v (float): previous first moment of var

    Returns:
        float, float: updated variable and the new moment
    """
    momentum = beta1 * v + (1 - beta1) * grad
    updated = var - alpha * momentum
    return updated, momentum

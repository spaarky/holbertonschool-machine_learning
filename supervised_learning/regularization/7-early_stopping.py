#!/usr/bin/env python3
"""_summary_
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Function that determines if you should stop gradient descent early

    Args:
        cost (float): current validation cost of the neural network
        opt_cost (float): lowest recorded validation cost of the neural network
        threshold (float): threshold used for early stopping
        patience (integer): patience count used for early stopping
        count (integer): count of how long the threshold has not been met

    Returns:
        (boolean): boolean of whether the network should be stopped early,
        (integer): updated count
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        return True, count
    return False, count

#!/usr/bin/env python3
"""Summary
"""


def moving_average(data, beta):
    """Function that calculates the weighted moving average of a data set

    Args:
        data (list): data to calculate the moving average
        beta (float): weight used for the moving average

    Returns:
        list: moving averages of data
    """
    if beta > 1 or beta < 0:
        return None
    vt = 0
    moving = []
    for i in range(len(data)):
        # Moving average:
        vt = beta * vt + (1 - beta) * data[i]
        # Correction of bias:
        correction = 1 - beta ** (i + 1)
        moving.append(vt / correction)
    return moving

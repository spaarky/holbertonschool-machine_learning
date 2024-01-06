#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def one_hot_decode(one_hot):
    """convert a one-hot matrix into a vector of labels

    Args:
        one_hot (numpy.ndarray): shape(classes, m), one-hot encoded matrix

    Returns:
        labels (numpy.ndarray): shape(m, ), containing the numeric
            label for each examples
    """

    # checks for value or type errors
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot) == 0 or len(one_hot.shape) is not 2:
        return None

    labels = np.argmax(one_hot, axis=0)
    return labels

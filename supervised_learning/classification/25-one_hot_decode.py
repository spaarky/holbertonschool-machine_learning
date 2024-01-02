#!/usr/bin/env python3
"""summary
"""
import numpy as np


def one_hot_decode(one_hot):
    """convert a one-hot matrix into a vector of labels

    Args:
        one_hot (numpy.ndarray): shape(classes, m), one-hot encoded matrix

    Returns:
        labels (numpy.ndarray): shape(m, ), containing the numeric label for each examples
    """

    labels = np.argmax(one_hot, axis=0)
    return labels

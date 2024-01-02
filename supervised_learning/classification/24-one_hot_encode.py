#!/usr/bin/env python3
"""summary
"""
import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a hot-one matrix

    Args:
        Y (numpy.ndarray): shape(m,), containing numeric class label
        classes (int): maximum number of classesfound in Y

    Returns:
        Y (numpy.ndarray): shape(classes, m), or None
    """

    try:
        encode = np.zeros((classes, Y.shape[0]))
        encode[Y, np.arange(Y.shape[0])] = 1
        return encode
    except Exception:
        return None

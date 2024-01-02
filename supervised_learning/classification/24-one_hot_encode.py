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

    num_instances = len(Y)
    one_hot_matrix = np.zeros((num_instances, classes))

    for i in range(num_instances):
        class_index = Y[i]
        one_hot_matrix[i, class_index] = 1

    return one_hot_matrix

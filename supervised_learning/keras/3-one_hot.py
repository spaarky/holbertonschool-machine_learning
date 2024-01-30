#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Function that converts a label vector into a one-hot matrix

    Args:
        labels (numpy.ndarray): labels of the data
        classes (integer, optional): number of classes. Defaults to None.

    Returns:
        (numpy.ndarray): one-hot matrix
    """

    return K.utils.to_categorical(labels, classes)

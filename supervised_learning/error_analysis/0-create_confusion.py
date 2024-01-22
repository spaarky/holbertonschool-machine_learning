#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that creates a confusion matrix

    Args:
        labels (numpy.ndarray): contains the correct labels for each
                                data points
        logits (numpy.ndarray): contains the predicted labels for each
                                data points

    Returns:
        (numpy.ndarray): the confusion matrix
    """
    return np.matmul(labels.T, logits)

#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def sensitivity(confusion):
    """Function to calculate the sensitivity of all classes

    Args:
        confusion (numpy.ndarray): confusion matrix

    Returns:
        (numpy.ndarray): sensitivity matrix
    """
    # sensitivity formula: TP / P
    TP = np.diagonal(confusion)
    P = np.sum(confusion, axis=1)
    return TP / P

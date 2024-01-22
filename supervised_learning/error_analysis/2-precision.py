#!/usr/bon/env python3
"""_summary_
"""
import numpy as np


def precision(confusion):
    """Function to calculate precision for all classes

    Args:
        confusion (mp.ndarray): confusion matrix

    Returns:
        (numpy.ndarray): precision matrix
    """
    # precision formula: TP / (TP + FP)
    TP = np.diagonal(confusion)
    P = np.sum(confusion, axis=0)
    return TP / P

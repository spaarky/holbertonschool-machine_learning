#!/usr/bin/env python3
"""_summary_
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Function to calculate f1 score

    Args:
        confusion (numpy.ndarray): confusion matrix

    Returns:
        (numpy.ndarray): f1 score for each class
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)
    return 2 * (prec * sens) / (prec + sens)

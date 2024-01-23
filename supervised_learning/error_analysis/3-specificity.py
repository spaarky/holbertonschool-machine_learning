#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def specificity(confusion):
    """Function to calculate the specificity

    Args:
        confusion (numpy.ndarray): confusion matrix

    Returns:
        (numpy.ndarray): specificity of each class in the confusion matrix
    """

    # Specificity = True Negative(TN) / Negatives(N); N = TN + False
    # Positive(FP)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    TNR = TN / (TN + FP)
    return TNR

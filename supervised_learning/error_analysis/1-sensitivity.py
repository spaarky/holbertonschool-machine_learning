#!/usr/bin/env python3
"""Summary
"""
import numpy as np


def sensitivity(confusion):

    # sensitivity formula: TP / P
    TP = np.diagonal(confusion)
    P = np.sum(confusion, axis=1)
    return TP / P

#!/usr/bin/env python3
"""Positional Encoding"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Summary"""
    PE = np.zeros((max_seq_len, dm))

    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            PE[i, j] = np.sin(i / (10000 ** (j / dm)))
            PE[i, j + 1] = np.cos(i / (10000 ** (j / dm)))

    return PE

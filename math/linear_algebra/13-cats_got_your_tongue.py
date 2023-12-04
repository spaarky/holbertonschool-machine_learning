#!/usr/bin/env python3
"""
    Module to concatenate 2 matrices along a certain axis
    using numpy
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Function that will concatenate 2 matrices along a certain
    axis using numpy"""
    return np.concatenate((mat1, mat2), axis)

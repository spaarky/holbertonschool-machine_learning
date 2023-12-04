#!/usr/bin/env python3
"""
    Module to slice a matrix
"""


def np_slice(matrix, axes={}):
    """Function that will slice a matrix"""
    sliced = []
    max_key = max(axes)
    for i in range(max_key + 1):
        if i in axes.keys():
            sliced.append(slice(*axes.get(i)))
        else:
            sliced.append(slice(None, None, None))
    return matrix[tuple(sliced)]

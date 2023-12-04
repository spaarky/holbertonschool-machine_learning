#!/usr/bin/env python3
"""
    Module to concatenate 2 matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function to concat 2 matrices"""
    if (len(mat1[0]) == len(mat2[0])) and axis == 0:
        concat = [element.copy() for element in mat1]
        concat += [element.copy() for element in mat2]
        return concat
    if (len(mat1) == len(mat2)) and axis == 1:
        concat = [mat1[index] + mat2[index] for index in range(len(mat1))]
        return concat
    else:
        return None

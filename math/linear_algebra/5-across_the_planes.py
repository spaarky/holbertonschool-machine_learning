#!/usr/bin/env python3
"""
    Module to add 2D matrices
"""


def add_matrices2D(mat1, mat2):
    """Function that will add 2 matrices"""
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    else:
        return [[(mat1[i][j] + mat2[i][j]) for i in range(len(mat1[0]))] for j in range(len(mat2))]

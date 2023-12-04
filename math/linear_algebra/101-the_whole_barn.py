#!/usr/bin/env python3
"""

"""


def add_matrices(mat1, mat2):
    """"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        return add_recursive(mat1, mat2)


def add_recursive(mat1, mat2):
    """"""
    if type(mat1[0]) != list:
        return [mat1[index] + mat2[index] for index in range(len(mat1))]
    else:
        result = []
        for index in range(len(mat1)):
            inner = add_recursive(mat1[index], mat2[index])
            result.append(inner)
        return result


def matrix_shape(matrix):
    """
    Needs a matrix as input
    Returns the shape as a list of integers
    """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])

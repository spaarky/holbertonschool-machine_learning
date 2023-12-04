#!/usr/bin/env python3
"""
    Module to calculate the sum of 2 matrices even 3D
"""


def add_matrices(mat1, mat2):
    """Main function to calculate the addition
    of 2 matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        return add_recursive(mat1, mat2)


def add_recursive(mat1, mat2):
    """Calculate the addition of 2 matrices
    using recursion"""
    if type(mat1[0]) != list:
        return [mat1[index] + mat2[index] for index in range(len(mat1))]
    else:
        result = []
        for index in range(len(mat1)):
            inner = add_recursive(mat1[index], mat2[index])
            result.append(inner)
        return result


def matrix_shape(matrix):
    """Return the shape of a matrix"""
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])

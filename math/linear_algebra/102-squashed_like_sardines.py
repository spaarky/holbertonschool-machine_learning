#!/usr/bin/env python3
"""
    Module to concatenate 2 matrices along an given axis
"""


def cat_matrices(mat1, mat2, axis=0):
    """Function that will return the concatenated matrix"""
    # Condition for 2 matrices to be concatenable
    if (len(matrix_shape(mat1)) > axis and len(matrix_shape(mat2)) > axis):
        return concat_recursive(mat1, mat2, axis)
    # Return None if the matrices aren't concatenable
    else:
        return None


def matrix_shape(matrix):
    """Function that will return the shape of a matrix
    as a list of int"""
    if type(matrix[0]) != list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape((matrix[0]))


def concat_recursive(mat1, mat2, axis):
    """Function that will return a new matrix resulting
    in the concatenation of 2 matrices"""
    result = []
    # When axis = 0, concatenate the matrices horizontaly
    if axis == 0:
        result = mat1 + mat2
        return result

    # When axis = 1, concatenate horizontaly but item by item
    for index in range(len(mat1)):
        result.append(concat_recursive(mat1[index], mat2[index], axis - 1))
    return result

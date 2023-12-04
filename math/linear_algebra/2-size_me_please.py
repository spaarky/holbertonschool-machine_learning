#!/usr/bin/env python3
def matrix_shape(matrix):
    """Function that will calculate the dimensions of a matrix"""
    shape = []
    shape.append(int(len(matrix)))
    shape.append(int(len(matrix[0])))
    if isinstance(matrix[0][0], list):
        shape.append(int(len(matrix[0][0])))
    return shape

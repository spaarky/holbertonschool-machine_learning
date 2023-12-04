#!/usr/bin/env python3
def matrix_shape(matrix):
    shape = []
    shape.append(len(matrix))
    shape.append(len(matrix[0]))
    if isinstance(matrix[0][0], list):
        shape.append(len(matrix[0][0]))
    return shape

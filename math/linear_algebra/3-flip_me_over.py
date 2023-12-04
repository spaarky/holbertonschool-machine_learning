#!/usr/bin/env python3
"""
    Module to calculate the transopose of a matrix
"""


def matrix_transpose(matrix):
    """Function that will calculate the transpose of a matrix"""
    transpose = []
    for row in range(len(matrix[0])):
        transpose.append([matrix[col][row] for col in range(len(matrix))])
    return transpose

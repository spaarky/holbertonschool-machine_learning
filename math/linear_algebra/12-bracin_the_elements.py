#!/usr/bin/env python3
"""
    Module to calculate the addition, substraction,
    multiplication and division of 2 matrices
    using numpy
"""


def np_elementwise(mat1, mat2):
    """Function to calculate te add, sub,
    mul and div of 2 matrices using numpy"""
    addition = mat1 + mat2
    substraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return addition, substraction, multiplication, division

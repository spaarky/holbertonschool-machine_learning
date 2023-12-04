#!/usr/bin/env python3
"""
    Module to multiply 2 matrices
"""


def mat_mul(mat1, mat2):
    """Function that will multiply 2 matrices"""
    # condition for 2 matrices to be multiplicable
    if (len(mat1[0]) == len(mat2)):
        new = []
        # for loop to get the row of mat1
        for row in range(len(mat1)):
            inner = []
            for col2 in range(len(mat2[0])):
                number = 0
                for col1 in range(len(mat1[0])):
                    number += (mat1[row][col1] * mat2[col1][col2])
                inner.append(number)
            new.append(inner)
        return new
    else:
        return None

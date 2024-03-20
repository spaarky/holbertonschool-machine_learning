#!/usr/bin/env python3
"""_summary_
"""


def determinant(matrix):
    """calculates the determinant of a matrix

    Args:
        matrix (list/nested list): matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    # recursive case: determinant of a 2x2 matrix
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    # 3x3 matrix using recursion
    first_row = matrix[0]
    det = 0
    cof = 1
    for i in range(len(matrix[0])):
        next_matrix = [x[:] for x in matrix]
        del next_matrix[0]
        for mat in next_matrix:
            del mat[i]
        det += first_row[i] * determinant(next_matrix) * cof
        cof = cof * -1

    return det


def cofactor(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_
    """
    if type(matrix) is not list or not len(matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a non-empty square matrix")
        if type(matrix[i]) is not list or not len(matrix[i]):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return [[1]]

    list_minor = []
    for i in range(len(matrix)):
        inner = []
        if i % 2 == 0:
            cof = 1
        else:
            cof = -1
        for j in range(len(matrix[0])):
            next_matrix = [x[:] for x in matrix]
            del next_matrix[i]
            for mat in next_matrix:
                del mat[j]
            det = determinant(next_matrix) * cof
            inner.append(det)
            cof = cof * -1
        list_minor.append(inner)

    return list_minor


def adjugate(matrix):
    """_summary_

    Args:
        matrix (_type_): _description_
    """
    adj = cofactor(matrix)
    transpose = []
    for j in range(len(adj[0])):
        inner = []
        for i in range(len(adj)):
            inner.append(adj[i][j])
        transpose.append(inner)
    return transpose

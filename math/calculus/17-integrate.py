#!/usr/bin/env python3
"""

"""


def poly_integral(poly, C=0):
    """"""
    if type(poly) is not list or poly == []:
        return None
    if type(C) is not int:
        return None

    # Find primitive
    for i in range(len(poly)):
        poly[i] = poly[i] / (i + 1)
    poly.insert(0, C)
    return poly

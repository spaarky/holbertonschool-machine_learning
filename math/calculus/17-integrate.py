#!/usr/bin/env python3
"""
    Module to calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """Function to calculate the integral of a polynomial"""
    if type(poly) is not list or poly == []:
        return None
    if type(C) is not int:
        return None

    # Find primitive
    for i in range(len(poly)):
        poly[i] = poly[i] / (i + 1)
    poly.insert(0, C)
    res = [int(x) if isinstance(x, float) and x.is_integer() else x for x in poly]
    return res

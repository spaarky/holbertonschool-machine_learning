#!/usr/bin/env python3
"""

"""


def poly_derivative(poly):
    """"""
    for i in range(len(poly)):
        poly[i] = poly[i] * i
    poly.pop(0)
    return poly

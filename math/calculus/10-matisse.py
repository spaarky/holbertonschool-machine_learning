#!/usr/bin/env python3
"""
    Module to derivate polynomials
"""


def poly_derivative(poly):
    """Function to derivate polynomials"""
    if type(poly) is not list or poly == []:
        return None
    if len(poly) < 1:
        return [0]

    for i in range(len(poly)):
        poly[i] = poly[i] * i
    poly.pop(0)
    return poly

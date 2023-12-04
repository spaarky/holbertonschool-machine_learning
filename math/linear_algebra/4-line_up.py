#!/usr/bin/env python3
"""
    Module to add arrays
"""


def add_arrays(arr1, arr2):
    """Function that will add 2 arrays and return the result"""
    res = []
    if len(arr1) != len(arr2):
        return None
    else:
        for index in range(len(arr1)):
            res.append(int(arr1[index]) + int(arr2[index]))
        return res

#!/usr/bin/env python3
"""
    Module to contatenate 2 arrays
"""


def cat_arrays(arr1, arr2):
    """Function that will contat 2 arrays"""
    res = []
    for index in range(len(arr1)):
        res.append(arr1[index])
    for index in range(len(arr2)):
        res.append(arr2[index])
    return res

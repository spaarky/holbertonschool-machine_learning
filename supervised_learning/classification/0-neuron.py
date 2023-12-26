#!/usr/bin/env python3

"""_summary_
"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """Initialization of a neuron

        Args:
            nx (integer): number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

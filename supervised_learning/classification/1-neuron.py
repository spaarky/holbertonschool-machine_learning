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
            nx (int): number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter function for private instance W

        Returns:
            int: Weights vector for the neuron
        """
        return self.__W

    @property
    def b(self):
        """Getter function for private instance b

        Returns:
            int: bias for the neuron
        """
        return self.__b

    @property
    def A(self):
        """Getter function for private instance A

        Returns:
            int: Activated output of the neuron (prediction)
        """
        return self.__A

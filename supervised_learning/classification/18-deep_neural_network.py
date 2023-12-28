#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a Deep Neural Network performing binary classification
    """
    def __init__(self, nx, layers):
        """Class constructor

        Args:
            nx (int): number of input features
            layers (list): list of number of nodes per hidden layers
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        # Number of layers in the neural network
        self.__L = len(layers)
        # dictionnary - holds all intermediary value of the network
        self.__cache = {}
        # dictionnary - holds all weights and bias of the network
        weights = {}
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            if i == 0:
                weights[key_w] = np.random.randn(layers[i], nx)*np.sqrt(2 / nx)
            else:
                weights[key_w] = np.random.randn(layers[i], layers[
                    i-1]) * np.sqrt(2 / layers[i-1])
            weights[key_b] = np.zeros((layers[i], 1))
        self.__weights = weights

    def forward_prop(self, X):
        """Calculates forward propagation of the neural network

        Args:
            X (numpy.ndarray): shape(nx, m), contains the input data

        Returns:
            The output of the neural network and the cache
        """

        # input layer
        self.__cache['A0'] = X

        # hidden and output layer
        for i in range(self.__L):
            # keys to acces weights and bias and store them in the cache
            key_w = 'W' + str(i + 1)  # key to get weight
            key_b = 'b' + str(i + 1)  # key to get bias
            key_cache = 'A' + str(i + 1)  # key to set the network output
            key_cache2 = 'A' + str(i)  # key to get the input

            # store activations in cache
            output_x = np.matmul(self.__weights[key_w],
                                 self.__cache[key_cache2])
            + self.__weights[key_b]
            output_a = 1 / (1 + np.exp(-output_x))
            self.__cache[key_cache] = output_a

        return output_a, self.__cache

    @property
    def cache(self):
        """Getter function for the private attribute cache

        Returns:
            dictionnary: hold all intermediary values of the network
        """

        return self.__cache

    @property
    def L(self):
        """Getter function for the private attribute L

        Returns:
            int: number of layers in the neural network
        """

        return self.__L

    @property
    def weights(self):
        """Getter function for private attribute weights

        Returns:
            dictionnary: holds all weights and bias of the network
        """

        return self.__weights

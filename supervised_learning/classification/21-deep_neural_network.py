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
        self.__L = len(layers)  # Number of layers
        self.__cache = {}  # holds all intermediary values
        weights = {}  # holds all weights and bias
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            # creates keys to access and set values in weights
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            # for input layer
            if i == 0:
                weights[key_w] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            # for hidden layer(s) and output layer
            else:
                weights[key_w] = np.random.randn(layers[i], layers[
                    i-1]) * np.sqrt(2 / layers[i-1])
            # bias initialized as a nul matrix
            weights[key_b] = np.zeros((layers[i], 1))
        self.__weights = weights

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): shape(1, m), contains the correct
                label for the input data
            cache (dictionnary): contains all the intermediary
                values of the network
            alpha (float, optional): learning rate. Defaults to 0.05.
        """

        m = Y.shape[1]
        for i in reversed(range(self.__L)):
            # creates the keys to access and store in cache
            key_w = 'W' + str(i + 1)  # key to get the weight
            key_b = 'b' + str(i + 1)  # key to get the bias
            key_cache = 'A' + str(i + 1)  # key to get the activated output
            key_cache_dw = 'A' + str(i)  # key to get the

            # activations
            A = cache[key_cache]
            A_dw = cache[key_cache_dw]
            # if it's the last hidden layer
            if i == self.__L - 1:
                dz = A - Y
                W = self.__weights[key_w]
            else:
                da = A * (1 - A)
                dz = np.matmul(W.T, dz)
                dz = dz * da
                W = self.__weights[key_w]
            dw = np.matmul(A_dw, dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights[key_w] = self.__weights[key_w] - alpha * dw.T
            self.__weights[key_b] = self.__weights[key_b] - alpha * db

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions

        Args:
            X (numpy.ndarrah): shape(nx, m), contains the input data
            Y (numpy.ndarray): shape(1, m), contains the correct label
                for the input data

        Returns:
            prediction (numpy.ndarray): shape(1, m), contains the predicted
                labels for each example
            cost (float): cost of the network
        """

        # get activation from forward propagation
        A, _ = self.forward_prop(X)
        # makes the prediction of the output
        # 1 when activation is greater or equal to .5 else 0
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def cost(self, Y, A):
        """Calculates the cost of the model using binary
            cross-entropy loss function

        Args:
            Y (numpy.ndarray): shape(1, m), contains the correct
                label for the input data
            A (numpy.ndarray): shape(1, m), contains the activated
                output of the neuron for each example

        Returns:
            float: cost if the model
        """

        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def forward_prop(self, X):
        """Calculates forward propagation of the neural network
        using sigmoid activationn function

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
            key_cache2 = 'A' + str(i)  # key to get the input of the layer

            # calculate and store activations in cache
            output_x = np.matmul(self.__weights[key_w], self.__cache[
                key_cache2]) + self.__weights[key_b]
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

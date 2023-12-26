#!/usr/bin/env python3

"""_summary_
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron by updating the private attributes W, b and A

        Args:
            X (numpy.ndarray): shape(nx, m), contains the input data
            Y (numpy.ndarray): shape(1, m), contains the correct
                              label for the input data
            iterations (int, optional): the number of iteration to
                                       train over. Defaults to 5000.
            alpha (float, optional): learning rate. Defaults to 0.05.
            verbose (bool, optional): print or not information about the
                                     training. Defaults to True.
            graph (bool, optional): print or not information about the training
                                    once its finished. Defaults to True.
            step (int, optional): step of iteration when to print informations.
                                 Defaults to 100.
        """

        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose is True and iterations % step == 0:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        cost = []
        for i in range(iterations + 1):
            activations = self.forward_prop(X)
            self.gradient_descent(X, Y, activations, alpha)
            cost.append(self.cost(Y, self.__A))
            if verbose is True and i % step == 0:
                print("Cost after {} iterations: {}"
                      .format(i, cost[i]))

        if graph is True:
            plt.plot(np.arange(0, iterations + 1), cost)
            plt.title("Training cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron

        Args:
            X (numpy.ndarray): shape(nx, m), contains the input data
                nx: number of input features
                m: number of examples
            Y (numpy.ndarray): shape(1, m), contains correct label
                              for the input data
            A (numpy.ndarray): shape(1, m), contains the activated
                              ouput of the neuron for each example
            alpha (float, optional): learning rate. Defaults to 0.05.
        """
        dz = A - Y
        dw = np.matmul(X, dz.T) / A.shape[1]
        db = np.sum(dz) / A.shape[1]
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def evaluate(self, X, Y):
        """Evaluates the neuron prediction

        Args:
            X (numpy.ndarray): shape(nx, m), contains the correct input data
                nx: number of input features
                m: number of examples
            Y (numpy.ndarray): shape(1, m)

        Returns:
            pred (numpy.ndarray): shape(1, m) contains the
                                 prediction label for each examples
            cost (integer): cost
        """
        self.forward_prop(X)
        pred = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return pred, cost

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): shape(1, m), contains the correct
                              label for the input data
            A (numpy.ndarray): shape(1, m), contains the activated
                              output of the neuron for each examples

        Returns:
            integer: cost
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def forward_prop(self, X):
        """Calculates the forward propagation of a single neuron

        Args:
            X (numpy.ndarray): shape of (nx, m) that contains the input data

        Returns:
            int: private attribute A result of the neuron
                activation using a sigmoid function
        """
        preactivation = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-preactivation))
        return self.__A

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

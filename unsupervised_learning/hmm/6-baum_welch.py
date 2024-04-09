#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """_summary_

    Args:
        Observation (_type_): _description_
        Emission (_type_): _description_
        Transition (_type_): _description_
        Initial (_type_): _description_

    Returns:
        _type_: _description_
    """

    N, M = Emission.shape
    T = Observation.shape[0]

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))

    for col in range(T - 2, -1, -1):
        for row in range(N):
            beta[row, col] = np.sum(beta[:, col + 1] *
                                    Transition[row, :] *
                                    Emission[:, Observation[col + 1]])

    return beta


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model

    Args:
        Observation (numpy.ndarray): shape (T,) that contains the index of
            the observation
        Emission (numpy.ndarray): shape (N, M) containing the emission
            probability of a specific observation given a hidden state
        Transition (numpy.ndarray): shape (N, N) containing the transition
            probabilities
        Initial (numpy.ndarray): shape (N, 1) containing the probability
            of starting in a particular hidden state

    Returns:
        P (float): the likelihood of the observations given the model
        F (numpy.ndarray): shape (N, T) containing the forward path
            probabilities
    """

    N, M = Emission.shape
    T = Observation.shape[0]

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial.T * Emission[:, Observation[0]]

    for col in range(1, T):
        for row in range(N):
            aux = alpha[:, col - 1] * Transition[:, row]
            alpha[row, col] = np.sum(aux * Emission[row, Observation[col]])

    return alpha


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """_summary_

    Args:
        Observations (_type_): _description_
        Transition (_type_): _description_
        Emission (_type_): _description_
        Initial (_type_): _description_
        iterations (int, optional): _description_. Defaults to 1000.
    """

    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    N, M = Emission.shape
    T = Observations.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    if iterations > 454:
        iterations = 454

    a = Transition.copy()
    b = Emission.copy()
    for n in range(iterations):
        alpha = forward(Observations, b, a, Initial.reshape((-1, 1)))
        beta = backward(Observations, b, a, Initial.reshape((-1, 1)))

        xi = np.zeros((N, N, T - 1))
        for col in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, col].T, a) *
                                 b[:, Observations[col + 1]].T,
                                 beta[:, col + 1])
            for row in range(N):
                numerator = alpha[row, col] * a[row, :] *\
                            b[:, Observations[col + 1]].T * beta[:, col + 1].T
                xi[row, :, col] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for k in range(M):
            b[:, k] = np.sum(gamma[:, Observations == k], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a, b


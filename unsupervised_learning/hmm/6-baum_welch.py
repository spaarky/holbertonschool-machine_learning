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

    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    sum_test = np.sum(Emission, axis=1)
    if not (sum_test == 1.0).all():
        return None, None
    sum_test = np.sum(Transition, axis=1)
    if not (sum_test == 1.0).all():
        return None, None
    sum_test = np.sum(Initial, axis=0)
    if not (sum_test == 1.0).all():
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))

    for col in range(T - 2, -1, -1):
        for row in range(N):
            beta[row, col] = np.sum(beta[:, col + 1] *
                                    Transition[row, :] *
                                    Emission[:, Observation[col + 1]])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])

    return P, beta


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

    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    sum_test = np.sum(Emission, axis=1)
    if not (sum_test == 1).all():
        return None, None
    sum_test = np.sum(Transition, axis=1)
    if not (sum_test == 1).all():
        return None, None
    sum_test = np.sum(Initial, axis=0)
    if not (sum_test == 1).all():
        return None, None

    N, _ = Emission.shape
    T = Observation.shape[0]

    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for col in range(1, T):
        for row in range(N):
            aux = F[:, col - 1] * Transition[:, row]
            F[row, col] = np.sum(aux * Emission[row, Observation[col]])

    P = np.sum(F[:, -1])

    return P, F


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """_summary_

    Args:
        Observations (_type_): _description_
        Transition (_type_): _description_
        Emission (_type_): _description_
        Initial (_type_): _description_
        iterations (int, optional): _description_. Defaults to 1000.
    """

    N, M = Emission.shape
    T = Observations.shape[0]

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial.T * Emission[:, Observations[0]]

    for col in range(1, T):
        for row in range(N):
            aux = alpha[:, col - 1] * Transition[:, row]
            alpha[row, col] = np.sum(aux * Emission[row, Observations[col]])

    return alpha


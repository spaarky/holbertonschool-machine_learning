#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


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

    N, M = Emission.shape
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

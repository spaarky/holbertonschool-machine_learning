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

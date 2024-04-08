#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
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

    N, _ = Emission.shape
    T = Observation.shape[0]
    if N != Transition.shape[0] or N != Transition.shape[1]:
        return None, None

    omega = np.zeros((N, T))
    aux = (Initial * Emission[:, Observation[0]].reshape(-1, 1))
    omega[:, 0] = aux.reshape(-1)

    backpointer = np.zeros((N, T))
    backpointer[:, 0] = 0
    for col in range(1, T):
        for row in range(N):
            prev = omega[:, col - 1]
            trans = Transition[:, row]
            em = Emission[row, Observation[col]]
            result = prev * trans * em
            omega[row, col] = np.amax(result)
            backpointer[row, col - 1] = np.argmax(result)

    path = []
    # Find the most probable last hidden state
    last_state = np.argmax(omega[:, T - 1])
    path.append(int(last_state))

    # backtracking algorithm gotten from first read
    for i in range(T - 2, -1, -1):
        path.append(int(backpointer[int(last_state), i]))
        last_state = backpointer[int(last_state), i]

    # Flip the path array since we were backtracking
    path.reverse()

    min_prob = np.amax(omega, axis=0)
    min_prob = np.amin(min_prob)

    return path, min_prob

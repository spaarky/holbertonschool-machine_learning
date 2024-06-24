#!/usr/bin/env python3
"""Summary"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Summary"""
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice([i for i in range(Q.shape[1])])
    else:
        action = np.argmax(Q[state, :])
    return action

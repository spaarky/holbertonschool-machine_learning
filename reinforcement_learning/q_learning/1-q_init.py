#!/usr/bin/env python3
"""Summary"""
import numpy as np


def q_init(env):
    """Summary"""
    Q_table = np.zeros((env.observation_space.n, env.action_space.n))
    return Q_table

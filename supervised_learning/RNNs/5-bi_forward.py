#!/usr/bin/env python3
"""summary"""
import numpy as np


class BidirectionalCell:
    """summary"""
    def __init__(self, i, h, o):
        """Class constructor"""
        self.Whf = np.random.normal(size=(h+i, h))
        self.Whb = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(2*h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step"""
        # First, concat h_prev and x_t to match the shape of Wh
        x_concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(x_concat, self.Whf) + self.bhf)

        return h_next

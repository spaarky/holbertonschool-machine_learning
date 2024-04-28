#!/usr/bin/env python3
"""summary"""
import numpy as np


class RNNCell:
    """summary"""
    def __init__(self, i, h, o):
        """Class constructor"""
        self.Wh = np.random.randn(h+i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step"""
        # First, concat h_prev and x_t to match the shape of Wh
        concat = np.concatenate((h_prev, x_t), axis=1)
        # Calculate the next hidden state
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        # Calculate the output of the cell
        y = np.matmul(h_next, self.Wy) + self.by
        # Apply the softmax activation function
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y

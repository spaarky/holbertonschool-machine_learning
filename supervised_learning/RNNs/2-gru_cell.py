#!/usr/bin/env python3
"""summary"""
import numpy as np


class GRUCell:
    """summary"""
    def __init__(self, i, h, o):
        """Class constructor"""
        self.Wz = np.random.normal(size=(h+i, h))
        self.Wr = np.random.normal(size=(h+i, h))
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step"""
        # First, concat h_prev and x_t to match the shape of Wh
        x_concat0 = np.concatenate((h_prev, x_t), axis=1)
        # Calculate the reset gate
        r = np.matmul(x_concat0, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))
        # Calculate the update gate
        z = np.matmul(x_concat0, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))

        # First, concat r * h_prev and x_t to match the shape of Wh
        x_concat1 = np.concatenate((r * h_prev, x_t), axis=1)
        # Calculate the candidate hidden state
        h = np.tanh(np.matmul(x_concat1, self.Wh) + self.bh)
        # Calculate the next hidden state
        h_next = (1 - z) * h_prev + z * h
        # Calculate the output of the cell
        y = np.matmul(h_next, self.Wy) + self.by
        # Apply the softmax activation function
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y

#!/usr/bin/env python3
"""summary"""
import numpy as np


class LSTMCell:
    """Summary"""
    def __init__(self, i, h, o):
        """Class constructor"""
        self.Wf = np.random.normal(size=(h+i, h))
        self.Wu = np.random.normal(size=(h+i, h))
        self.Wc = np.random.normal(size=(h+i, h))
        self.Wo = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """summary"""
        # First, concat h_prev and x_t to match the shape of Wh
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        # Calculate the forget gate
        f = np.matmul(x_concat, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))
        # Calculate the update gate
        u = np.matmul(x_concat, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))
        # Calculate the candidate hidden state
        c = np.tanh(np.matmul(x_concat, self.Wc) + self.bc)
        # Calculate the next cell state
        c_next = f * c_prev + u * c

        # Calculate the output gate
        o = np.matmul(x_concat, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))

        # Calculate the next hidden state
        h_next = o * np.tanh(c_next)
        # Calculate the output of the cell
        y = np.matmul(h_next, self.Wy) + self.by
        # Apply the softmax activation function
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y

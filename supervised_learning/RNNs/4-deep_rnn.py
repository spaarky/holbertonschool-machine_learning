#!/usr/bin/env python3
"""summary"""
import numpy as np


def deep_rnn(rnn_cells, X, h_o):
    """summary"""
    t, m, i = X.shape
    l, m, h = h_o.shape
    o = rnn_cells[-1].Wy.shape[1]
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_o

    for i in range(t):
        h_aux = X[i]

        for j in range(len(rnn_cells)):
            r_cell = rnn_cells[j]
            x_t = h_aux
            h_prev = H[i, j]
            h_next, y_next = r_cell.forward(h_prev, x_t)
            h_aux = h_next
            H[i + 1, j] = h_aux

        Y[i] = y_next

    return H, Y

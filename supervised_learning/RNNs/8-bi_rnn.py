#!/usr/bin/env python3
"""summary"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """summary"""
    t, m, i = X.shape
    h = h_0.shape[1]
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))
    h_ft = h_0
    h_bt = h_t

    for step in range(t):
        x_ft = X[step]
        x_bt = X[-(step - 1)]

        h_ft = bi_cell.forward(h_ft, x_ft)
        h_bt = bi_cell.backward(h_bt, x_bt)

        H_forward[step] = h_ft
        H_backward[-(step + 1)] = h_bt

    H = np.concatenate((H_forward, H_backward), axis=-1)
    Y = bi_cell.output(H)

    return H, Y

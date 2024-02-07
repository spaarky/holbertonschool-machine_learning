#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Function that performs forward propagation over a convolutional
        layer of a neural network

    Args:
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        W (numpy.ndarray): of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution
        b (numpy.ndarray): of shape (1, 1, 1, c_new) containing the
            biases applied to the convolution
        activation (string): activation function applied to the convolution
        padding (string, optional): is either same or valid, indicating the
            type of padding used. Defaults to "same".
        stride (tuple, optional): of (sh, sw) containing the strides for the
            convolution. Defaults to (1, 1).

    Returns:
        (numpy.ndarray): containing the output of the convolutional layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    pad_h, pad_w = 0, 0
    if padding == 'same':
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    padded = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode='constant', constant_values=(0, 0))

    conv_h = int((h_prev + 2 * pad_h - kh) / sh + 1)
    conv_w = int((w_prev + 2 * pad_w - kw) / sw + 1)
    convolved = np.zeros((m, conv_h, conv_w, c_new))

    # row = height, col = width
    for row in range(conv_h):
        for col in range(conv_w):
            for ch in range(c_new):
                slice = padded[:, row * sh:row * sh + kh, col * sw:col
                               * sw + kw]
                slice_sum = np.sum(slice * W[:, :, :, ch], axis=(1, 2, 3))
                convolved[:, row, col, ch] = slice_sum

    return activation(convolved + b)

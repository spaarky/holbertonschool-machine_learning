#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs forward propagation over a pooling
        layer of a neural network

    Args:
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        kernel_shape (tuple): of (kh, kw) containing the size of the
            kernel for the pooling
        stride (tuple, optional): of (sh, sw) containing the strides
            for the convolution. Defaults to (1, 1).
        mode (string, optional): containing either max or avg, indicating
            whether to perform maximum or average pooling, respectively.
            Defaults to 'max'.

    Returns:
        (numpy.ndarray): containing the output of the pooling layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pool_h = (h_prev - kh) // sh + 1
    pool_w = (w_prev - kw) // sw + 1

    pooled = np.zeros((m, pool_h, pool_w, c_prev))
    # row = height, col = width
    for row in range(pool_h):
        for col in range(pool_w):
            slice_A = A_prev[:, row * sh:row * sh + kh, col * sw:col * sw + kh]
            if mode == "max":
                pooled[:, row, col] = np.max(slice_A, axis=(1, 2))
            if mode == "avg":
                pooled[:, row, col] = np.mean(slice_A, axis=(1, 2))
    return pooled

#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs back propagation over a pooling layer of a
        neural network

    Args:
        dA (numpy.ndarray): shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the output of the pooling
            layer
        A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c) containing the
            output of the previous layer
        kernel_shape (tuple): (kh, kw) containing the size of the kernel for
            the pooling
        stride (tuple, optional): (sh, sw) containing the strides for the
            pooling. Defaults to (1, 1).
        mode (string, optional): containing either max or avg, indicating
            whether to perform maximum or average pooling, respectively.
            Defaults to 'max'.

    Returns:
        (numpy.ndarray): partial derivatives with respect to the previous
            layer (dA_prev)
    """

    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    # row = height, col = width
    for img in range(m):
        A_img = A_prev[img]
        for row in range(h_new):
            for col in range(w_new):
                for ch in range(c_new):
                    row_start = row * sh
                    row_end = row * sh + kh
                    col_start = col * sw
                    col_end = col * sw + kw

                    slice_A = A_img[row_start:row_end, col_start:col_end, ch]
                    if mode == "max":
                        mask = (slice_A == np.max(slice_A)).astype(int)
                        aux = dA[img, row, col, ch] * mask
                        dA_prev[img, row_start:row_end, col_start:col_end,
                                ch] += aux
                    if mode == "avg":
                        average = dA[img, row, col, ch] / (kh * kw)
                        mask = np.ones(kernel_shape) * average
                        dA_prev[img, row_start:row_end, col_start:col_end,
                                ch] += mask
    return dA_prev

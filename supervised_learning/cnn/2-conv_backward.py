#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over a convolutional layer
        of a neural network

    Args:
        dZ (numpy.ndarray): shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the unactivated output
            of the convolutional layer
        A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
        W (numpy.ndarray): shape (kh, kw, c_prev, c_new) containing the
            kernels for the convolution
        b (numpy.ndarray): shape (1, 1, 1, c_new) containing the biases
            applied to the convolution
        padding (string, optional): is either same or valid, indicating the
            type of padding used. Defaults to "same".
        stride (tuple, optional): (sh, sw) containing the strides for the
            convolution. Defaults to (1, 1).

    Returns:
        (numpy.ndarray): containing the partial derivatives with respect to
            the previous layer (dA_prev), the kernels (dW), and the biases
            (db), respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    pad_h, pad_w = (0, 0)
    if padding == "same":
        pad_h = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pad_w = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    A_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                   mode="constant", constant_values=(0, 0))

    dA_pad = np.pad(dA, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    mode="constant", constant_values=(0, 0))

    # row = height, col = width
    for img in range(m):
        A_img = A_pad[img]
        dA_img = dA_pad[img]
        for row in range(h_new):
            for col in range(w_new):
                for ch in range(c_new):
                    # corners of the slice
                    row_start = row * sh
                    row_end = row * sh + kh
                    col_start = col * sw
                    col_end = col * sw + kw

                    slice_A = A_img[row_start:row_end, col_start:col_end, :]
                    aux = W[:, :, :, ch] * dZ[img, row, col, ch]
                    dA_img[row_start:row_end, col_start:col_end] += aux
                    dW[:, :, :, ch] += slice_A * dZ[img, row, col, ch]
        if padding == "same":
            dA[img, :, :, :] += dA_img[pad_h: -pad_h, pad_w: - pad_w]
        if padding == "valid":
            dA[img, :, :, :] += dA_img
    return dA, dW, db

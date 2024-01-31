#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function that performs a valid convolution on grayscale images

    Args:
        images (numpy.ndarray): matrix of shape (m, h, w) containing multiple
        kernel (numpy.ndarray): shape (kh, kw) containing the kernel

    Returns:
        convolved (numpy.ndarray): convolved images
    """

    c_images = images.shape[0]
    f_height = kernel.shape[0]
    f_width = kernel.shape[1]

    # set padding if filter dimensions are odd or even
    if f_height % 2 != 0:
        padding_h = (f_height - 1) // 2
    else:
        padding_h = f_height // 2
    if f_width % 2 != 0:
        padding_w = (f_width - 1) // 2
    else:
        padding_w = f_width // 2

    c_height = images.shape[1]
    c_width = images.shape[2]
    pad_images = (np.pad(images, ((0, 0), (padding_h, padding_h),
                                  (padding_w, padding_w))))
    convolved = np.zeros((c_images, c_height, c_width))
    for row in range(c_height):
        for col in range(c_width):
            mul_ele = (pad_images[:, row:row + f_height, col:col + f_width]
                       * kernel)
            sum_ele = np.sum(mul_ele, axis=(1, 2))
            convolved[:, row, col] = sum_ele
    return convolved

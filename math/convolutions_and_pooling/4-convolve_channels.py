#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Function that performs a convolution on images with channels

    Args:
        images (numpy.ndarray): matrix of shape (m, h, w) containing multiple
        kernel (numpy.ndarray): shape (kh, kw) containing the kernel
        padding (string / tuple, optional): either a tuple of (ph, pw),
            'same', or 'valid'. Defaults to 'same'.
        stride (tuple, optional): tuple of (sh, sw). Defaults to (1, 1).

    Returns:
        convolved (numpy.ndarray): convolved images
    """

    # get the number of images, their height and width.
    # and kernel height and width.
    c_images, images_h, images_w, _ = images.shape
    f_height = kernel.shape[0]
    f_width = kernel.shape[1]
    # get stride from tuple
    stride_h, stride_w = stride

    # set padding dimensions
    if padding == "same":
        padding_h = ((images_h - 1) * stride_h + f_height - images_h) // 2 + 1
        padding_w = ((images_w - 1) * stride_w + f_width - images_w) // 2 + 1
    elif padding == "valid":
        padding_h, padding_w = (0, 0)
    else:
        padding_h, padding_w = padding

    # set convolved images dimensions
    c_height = (images.shape[1] + 2 * padding_h - f_height) // stride_h + 1
    c_width = (images.shape[2] + 2 * padding_w - f_width) // stride_w + 1
    pad_images = (np.pad(images, ((0, 0), (padding_h, padding_h),
                                  (padding_w, padding_w), (0, 0))))

    convolved = np.zeros((c_images, c_height, c_width))
    for row in range(c_height):
        for col in range(c_width):
            pad_ele = pad_images[:, row * stride_h:row * stride_h + f_height,
                                 col * stride_w:col * stride_w + f_width]
            sum_mul_ele = np.sum(pad_ele * kernel, axis=(1, 2, 3))
            convolved[:, row, col] = sum_mul_ele
    return convolved

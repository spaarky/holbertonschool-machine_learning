#!/usr/bin/env python3
"""_summary_
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images

    Args:
        images (numpy.ndarray): matrix of shape (m, h, w) containing multiple
        kernel (numpy.ndarray): shape (kh, kw) containing the kernel
        stride (tuple, optional): tuple of (sh, sw). Defaults to (1, 1).
        mode (str, optional): indicates the type of pooling. Defaults to 'max'.

    Returns:
        pooled (numpy.ndarray): contains the pooled images
    """
    c_images, images_h, images_w, channels = images.shape
    f_height = kernel_shape[0]
    f_width = kernel_shape[1]
    stride_h, stride_w = stride

    p_height = (images_h - f_height) // stride_h + 1
    p_width = (images_w - f_width) // stride_w + 1

    pooled = np.zeros((c_images, p_height, p_width, channels))
    for row in range(p_height):
        for col in range(p_width):
            ele = images[:, row * stride_h:row * stride_h + f_height,
                         col * stride_w:col * stride_w + f_width]

            if mode == "max":
                pooled[:, row, col] = np.max(ele, axis=(1, 2))
            if mode == "avg":
                pooled[:, row, col] = np.mean(ele, axis=(1, 2))

    return pooled

#!/usr/bin/env python3
"""summary
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction

    Args:
        y (tf.placeholder): contains the labels of the input data
        y_pred (tf.tensor): contains the network's prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)

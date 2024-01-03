#!/usr/bin/env python3
"""summary
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction

    Args:
        y (tf.placeholder): contains the input data
        y_pred (tf.tensor): contains the network's prediction
    """
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    return mean

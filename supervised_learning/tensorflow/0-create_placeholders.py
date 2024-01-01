#!/usr/bin/env python3
"""summary
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """Function that return two placeholders for the neural network

    Args:
        nx (int): number of feature columns in the classifier
        classes (int): number of classes in the classifier

    Returns:
        classifier: x and y
    """

    x = tf.placeholder(float, shape=[None, nx], name='x')
    y = tf.placeholder(float, shape=[None, classes], name='y')
    return x, y

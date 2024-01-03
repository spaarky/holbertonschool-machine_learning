#!/usr/bin/env python3
"""summary
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Function that creates the training operation for the network

    Args:
        loss (float): loss of the network's prediction
        alpha (float): learning rate
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)

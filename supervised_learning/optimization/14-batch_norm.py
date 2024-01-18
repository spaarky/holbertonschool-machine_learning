#!/usr/bin/env python3
"""Summary
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization layer for a
        neural network in tensorflow

    Args:
        prev (float): activated output of the previous layer
        n (int): number of nodes in the layer to be created
        activation (str): activation function that should be
            used on the output of the layer

    Returns:
        : tensor of the activated output for the layer
    """
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layerX = tf.layers.dense(prev, n, kernel_initializer=weights)
    mean, variance = tf.nn.moments(layerX, 0)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(
        layerX, mean, variance, beta, gamma, epsilon
    )
    return activation(batch_norm)

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
    weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.dense(units=n, kernel_initializer=weights, name="layer")
    X = layer(prev)
    mean, variance = tf.nn.moments(X, axis=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(x=X,
                                           mean=mean,
                                           variance=variance,
                                           offset=beta,
                                           scale=gamma,
                                           variance_epsilon=epsilon)
    return activation(batch_norm)

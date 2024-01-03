#!/usr/bin/env python3
"""summary
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Write a function that return the output of the layer

    Args:
        prev (float): tensor output of the previous layer
        n (integer): number of nodes in the layer to create
        activation (function): activation function to use in the layer

    Returns:
        ?: tensor output of the layer
    """

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init, name='layer')

    return layer(prev)

#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function

    Args:
        prev (tensor): tensor containing the output of the previous layer
        n (integer): number of nodes the new layer should contain
        activation (string): activation function that should be used
            on the layer
        keep_prob (float): probability that a node will be kept

    Returns:
        (float): output of the new layer
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)

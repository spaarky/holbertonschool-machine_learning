#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tensorflow layer that includes L2 regularization

    Args:
        prev (tensor): tensor containing the output of the previous layer
        n (integer): number of nodes the new layer should contain
        activation (string): activation function that should be used
            on the layer
        lambtha (float): L2 regularization parameter

    Returns:
        float: output of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    regu = tf.keras.regularizers.l2(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regu,
                            name="layer")
    return layer(prev)

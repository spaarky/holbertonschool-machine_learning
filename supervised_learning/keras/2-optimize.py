#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Function that sets up Adam optimization for a keras model with
        categorical crossentropy loss and accuracy metrics

    Args:
        network (): model to optimize
        alpha (float): learning rate
        beta1 (float): first Adam optimization parameter
        beta2 (float): second Adam optimization parameter

    Returns:
        None
    """
    optimizer = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    loss = 'categorical_crossentropy'
    network.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return None

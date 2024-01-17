#!/usr/bin/env python3
"""Summary
"""


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Function that creates the training operation for a neural network in
        tensorflow using the RMSProp optimization algorithm

    Args:
        loss (float): loss of the network
        alpha (float): learning rate
        beta2 (float): RMSprop weight
        epsilon (float): small number to avoid division by 0

    Returns:
        _type_: RMSprop optimization operation
    """
    rms = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return rms.minimize(loss)

#!/usr/bin/env python3
"""Summary
"""


def create_momentum_op(loss, alpha, beta1):
    """Function creates the training operation for a neural network in
        tensorflow using the gradient descent with momentum
        optimization algorithm

    Args:
        loss (float): loss of the network
        alpha (float): learning rate
        beta1 (float): momentum rate

    Returns:
        _type_: momentum optimization operation
    """
    return tf.train.MomentumOptimizer(alpha, momentum=beta1).minimize(loss)

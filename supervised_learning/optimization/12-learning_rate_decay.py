#!/usr/bin/env python3
"""Summary
"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that creates a learning rate decay operation in
        tensorflow using inverse time decay

    Args:
        alpha (float): learning rate
        decay_rate (float): weight used to determine the rate at
            which alpha will decay
        global_step (int): number of passes of gradient descent
            that have elapsed
        decay_step (int): number of passes of gradient descent that
            should occur before alpha is decayed further

    Returns:
        : learning rate decay operation
    """
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
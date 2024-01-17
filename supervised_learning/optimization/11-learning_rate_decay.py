#!/usr/bin/env python3
"""Summary
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that updates the learning rate using inverse
        time decay in numpy

    Args:
        alpha (float): learning rate
        decay_rate (float): weight used to determine the rate at
            which alpha will decay
        global_step (int): number of passes of gradient descent
            that have elapsed
        decay_step (int): number of passes of gradient descent that
            should occur before alpha is decayed further

    Returns:
        float: updated value for alpha
    """
    epoch = global_step // decay_step
    new_alpha = alpha / (1 + decay_rate * epoch)
    return new_alpha

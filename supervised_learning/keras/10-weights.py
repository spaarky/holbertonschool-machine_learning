#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Function that saves a model’s weights

    Args:
        network (object): model whose weights should be saved
        filename (string): path of the file that the weights should be
            saved to
        save_format (string, optional): format in which the weights should be
            saved. Defaults to 'h5'.

    Returns:
        None
    """
    network.save_weights(
        filepath=filename,
        overwrite=True,
        save_format=save_format
    )
    return None


def load_weights(network, filename):
    """Function that loads a model’s weights

    Args:
        network (object): model to which the weights should
            be loaded
        filename (string): path of the file that the weights should
            be loaded from

    Returns:
        None
    """
    network.load_weights(
        filepath=filename,
        by_name=False
    )
    return None

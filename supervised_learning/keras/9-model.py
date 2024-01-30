#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def save_model(network, filename):
    """Function that saves an entire model

    Args:
        network (object): model to save
        filename (string): path of the file that the model should be
            saved to

    Returns:
        None
    """
    network.save(
        filepath=filename,
        overwrite=True,
        include_optimizer=True
    )
    return None


def load_model(filename):
    """Function that loads an entire model

    Args:
        filename (string): path of the file that the model should be
            loaded from

    Returns:
        (object): the loaded model
    """
    network = K.models.load_model(
        filepath=filename,
        custom_objects=None,
        compile=True
    )
    return network

#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model’s configuration in JSON format

    Args:
        network (object): model whose configuration should be saved
        filename (string): path of the file that the configuration
            should be saved to

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """Function that loads a model with a specific configuration

    Args:
        filename (string): path of the file that the configuration
            should be saved to

    Returns:
        (object): the loaded model
    """
    with open(filename, 'r') as f:
        config = f.read()
    network = K.models.model_from_json(config)
    return network

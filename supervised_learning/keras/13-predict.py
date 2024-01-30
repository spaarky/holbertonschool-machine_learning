#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Function that makes a prediction using a neural network

    Args:
        network (object): network model to make the prediction with
        data (numpy.ndarray): input data to make the prediction with
        verbose (boolean, optional): determines if output should be
            printed during the testing process. Defaults to False.

    Returns:
        _type_: _description_
    """
    return network.predict(x=data, batch_size=None, verbose=verbose)

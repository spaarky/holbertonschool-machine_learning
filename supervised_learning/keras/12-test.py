#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Function that tests a neural network

    Args:
        network (object): network model to test
        data (numpy.ndarray): input data to test the model with
        labels (numpy.ndarray): correct one-hot labels of data
        verbose (boolean, optional): determines if output should be
            printed during the testing process. Defaults to True.

    Returns:
        (floats): loss and accuracy of the model with the testing data
    """
    return network.evaluate(x=data, y=labels,
                            batch_size=None, verbose=verbose)

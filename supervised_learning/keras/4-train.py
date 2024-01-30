#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Function that trains a model using mini-batch gradient descent

    Args:
        network (): model to train
        data (numpy.ndarray): shape (m, nx) containing the input data
        labels (numpy.ndarray): shape (m, classes) containing
            the labels of data
        batch_size (integer): size of the batch used for
            mini-batch gradient descent
        epochs (integer): number of passes through data for
            mini-batch gradient descent
        verbose (bool, optional): boolean that determines if output should be
            printed during training. Defaults to True.
        shuffle (bool, optional): boolean that determines whether to shuffle
            the batches every epoch. Defaults to False.

    Returns:
        object: History object generated after training the model
    """

    History = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, shuffle=shuffle)

    return History

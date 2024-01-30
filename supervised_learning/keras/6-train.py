#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
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
        validatrion_data (numpy.ndarray, optional): data to validate
            the model with.
        early_stopping (bool, optional): boolean that indicates whether
            early stopping should be used. Defaults to False.
        patience (integer, optional): patience used for early stopping.
        verbose (bool, optional): boolean that determines if output should be
            printed during training. Defaults to True.
        shuffle (bool, optional): boolean that determines whether to shuffle
            the batches every epoch. Defaults to False.

    Returns:
        object: History object generated after training the model
    """
    callback = []
    if early_stopping and validation_data:
        callback.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience))

    History = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callback)

    return History

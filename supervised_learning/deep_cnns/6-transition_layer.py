#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """_summary_

    Args:
        X (_type_): _description_
        nb_filters (_type_): _description_
        compression (_type_): _description_
    """
    init = K.initializers.he_normal()
    n_number = int(nb_filters * compression)

    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(filters=n_number, kernel_size=1, strides=1,
                            padding='same', kernel_initializer=init)(act1)

    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                         padding='same')(conv1)

    return avg_pool, n_number

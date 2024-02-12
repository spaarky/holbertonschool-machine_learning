#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """_summary_

    Args:
        X (_type_): _description_
        nb_filters (_type_): _description_
        growth_rate (_type_): _description_
        layers (_type_): _description_
    """
    init = K.initializers.he_normal()

    for i in range(layers):
        # 1x1 convolution
        norm1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation('relu')(norm1)
        conv1 = K.layers.Conv2D(filters=4*growth_rate, kernel_size=1,
                                padding='same', kernel_initializer=init)(act1)

        # 3x3 convolution
        norm2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation('relu')(norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                padding='same', kernel_initializer=init)(act2)

        X = K.layers.Concatenate()([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters

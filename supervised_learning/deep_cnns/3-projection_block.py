#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds an identity block as described in
        Deep Residual Learning for Image Recognition (2015)

        Args:
            A_prev (tf.Tensor): The output from the previous layer.
            filters (tuple): A tuple of F11, F3 and F12, respectively.
            F11 is the number of filters in the first 1x1 convolution.
            F3 is the number of filters in the 3x3 convolution.
            F12 is the number of filters in the second 1x1 convolution

        Returns:
            Activated output of the identity block
    """
    init = K.initializers.he_normal()
    F11, F3, F12 = filters

    # first layer
    conv0 = K.layers.Conv2D(F11, kernel_size=1, padding='same', strides=s,
                            kernel_initializer=init)(A_prev)
    norm0 = K.layers.BatchNormalization()(conv0)
    act0 = K.layers.Activation('relu')(norm0)

    # second layer
    conv1 = K.layers.Conv2D(F3, kernel_size=3, padding='same', strides=1,
                            kernel_initializer=init)(act0)
    norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(norm1)

    # third layer
    conv2 = K.layers.Conv2D(F12, kernel_size=1, padding='same', strides=1,
                            kernel_initializer=init)(act1)
    norm2 = K.layers.BatchNormalization()(conv2)

    # shortcut
    shortcut = K.layers.Conv2D(F12, kernel_size=1, padding='same', strides=s,
                               kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization()(shortcut)
    X = K.layers.Add()([norm2, shortcut])

    act3 = K.layers.Activation('relu')(X)

    return act3

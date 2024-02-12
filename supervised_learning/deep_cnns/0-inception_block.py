#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block as described in Going Deeper with
    Convolutions (2014).

    Args:
        A_prev (tf.Tensor): The output from the previous layer.
        filters (tuple): A tuple of F1, F3R, F3, F5R, F5, and FPP,
            respectively. F1 is the number of filters in the 1x1
            convolution. F3R is the number of filters in the 1x1
            convolution before the 3x3 convolution. F3 is the number of
            filters in the 3x3 convolution. F5R is the number of filters
            in the 1x1 convolution before the 5x5 convolution. F5 is the
            number of filters in the 5x5 convolution. FPP is the number
            of filters in the 1x1 convolution after the max pooling.

    Returns:
        tf.Tensor: The concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv1x1 = K.layers.Conv2D(F1, (1, 1), padding='same',
                              activation='relu')(A_prev)

    # 1x1 convolution before 3x3 convolution
    conv3x3 = K.layers.Conv2D(F3R, (1, 1), padding='same',
                              activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(F3, (3, 3), padding='same',
                              activation='relu')(conv3x3)

    # 1x1 convolution before 5x5 convolution
    conv5x5 = K.layers.Conv2D(F5R, (1, 1), padding='same',
                              activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                              activation='relu')(conv5x5)

    # Max pooling
    max_pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                     padding='same')(A_prev)
    max_pool = K.layers.Conv2D(FPP, (1, 1), padding='same',
                               activation='relu')(max_pool)

    # Filter concatenation
    output = K.layers.concatenate([conv1x1, conv3x3, conv5x5, max_pool],
                                  axis=-1)

    return output

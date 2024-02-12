#!/usr/bin/env python3
"""_summary_
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """_summary_

    Args:
        growth_rate (int, optional): _description_. Defaults to 32.
        compression (float, optional): _description_. Defaults to 1.0.
    """
    init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    n_filters = 2 * growth_rate

    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(filters=n_filters, kernel_size=7,
                            strides=2, kernel_initializer=init,
                            padding='same')(act1)

    max_pool1 = K.layers.MaxPooling2D(pool_size=3, strides=2)(conv1)

    # Dense block 1
    dense_output_1, n_filters = dense_block(max_pool1, n_filters, growth_rate, 6)

    # Transition layer 1
    transition_output_1, n_filters = transition_layer(dense_output_1, n_filters, compression)

    # Dense block 2
    dense_output_2, n_filters = dense_block(transition_output_1, n_filters, growth_rate, 12)

    # Transition layer 2
    transition_output_2, n_filters = transition_layer(dense_output_2, n_filters, compression)

    # Dense block 3
    dense_output_3, n_filters = dense_block(transition_output_2, n_filters, growth_rate, 24)

    # Transition layer 3
    transition_output_3, n_filters = transition_layer(dense_output_3, n_filters, compression)

    # Dense block 4
    dense_output_4, n_filters = dense_block(transition_output_3, n_filters, growth_rate, 16)

    avg_pool_1 = K.layers.AveragePooling2D(pool_size=7, strides=1, padding='valid')(dense_output_4)

    softmax = K.layers.Dense(units=1000, activation='softmax', kernel_initializer=init)(avg_pool_1)

    model = K.Model(inputs=X, outputs=softmax)

    return model

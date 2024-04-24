#!/usr/bin/env python3
"""summary
"""
import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """_summary_

    Args:
        input_dims (_type_): _description_
        hidden_layers (_type_): _description_
        latent_dims (_type_): _description_
        lambtha (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))
    sparsity = keras.regularizers.l1(lambtha)

    # Encoder model
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activity_regularizer=sparsity,
                                 activation='relu')(input_encoder)
    for enc in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[enc],
                                     activity_regularizer=sparsity,
                                     activation='relu')(encoded)

    # Latent layer
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    # Decoded model
    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)
    for dec in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[dec],
                                     activation='relu')(decoded)
    last = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    encoder_output = encoder(input_encoder)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(inputs=input_encoder, outputs=decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

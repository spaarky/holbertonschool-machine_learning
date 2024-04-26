#!/usr/bin/env python3
"""Summary
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """summary
    """
    input_encoder = keras.layers.Input(shape=(input_dims,))
    output_encoder = input_encoder
    for units in hidden_layers:
        output_encoder = keras.layers.Dense(units,
                                            activation='relu')(output_encoder)
    output_encoder = keras.layers.Dense(latent_dims,
                                        activation='relu')(output_encoder)
    encoder = keras.models.Model(input_encoder, output_encoder)

    input_decoder = keras.layers.Input(shape=(latent_dims,))
    output_decoder = input_decoder
    for units in reversed(hidden_layers):
        output_decoder = keras.layers.Dense(units,
                                            activation='relu')(output_decoder)
    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(output_decoder)
    decoder = keras.models.Model(input_decoder, output_decoder)

    output_autoencoder = decoder(encoder(input_encoder))
    autoencoder = keras.models.Model(input_encoder, output_autoencoder)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder

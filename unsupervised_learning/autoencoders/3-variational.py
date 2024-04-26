#!/usr/bin/env python3
""" summary """
import tensorflow.keras as K
import tensorflow as tf


def sampling(args):
    """summary
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """summary
    """
    encoder_input = K.layers.Input(shape=(input_dims,))
    enc_hidden = encoder_input
    for nodes in hidden_layers:
        enc_hidden = K.layers.Dense(nodes, activation='relu')(enc_hidden)
    z_mean = K.layers.Dense(latent_dims, activation=None)(enc_hidden)
    z_log_var = K.layers.Dense(latent_dims, activation=None)(enc_hidden)

    z = K.layers.Lambda(sampling,
                        output_shape=(latent_dims,))([z_mean, z_log_var])

    encoder = K.models.Model(encoder_input,
                             [z_mean, z_log_var, z], name='encoder')

    latent_input = K.layers.Input(shape=(latent_dims,))
    dec_hidden = latent_input
    for nodes in reversed(hidden_layers):
        dec_hidden = K.layers.Dense(nodes, activation='relu')(dec_hidden)
    decoder_output = K.layers.Dense(input_dims,
                                    activation='sigmoid')(dec_hidden)

    decoder = K.models.Model(latent_input, decoder_output, name='decoder')

    autoencoder_output = decoder(encoder(encoder_input)[2])
    autoencoder = K.models.Model(encoder_input,
                                 autoencoder_output, name='autoencoder')

    reconstruction_loss = K.losses.binary_crossentropy(
        encoder_input, autoencoder_output)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    autoencoder.add_loss(vae_loss)

    autoencoder.compile(optimizer='adam')

    return encoder, decoder, autoencoder

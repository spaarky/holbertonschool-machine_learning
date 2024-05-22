#!/usr/bin/env python3
"""Summary"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ Class RNNDecoder to decode for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """Summary"""
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding)

        self.gru = tf.keras.layers.GRU(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True)

        self.F = tf.keras.layers.Dense(vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Summary"""
        context, _ = self.attention(s_prev, hidden_states)

        x = self.embedding(x)

        context_expanded = tf.expand_dims(context, axis=1)
        inputs = tf.concat([context_expanded, x], axis=-1)

        outputs, s = self.gru(inputs=inputs)

        outputs_reshaped = tf.reshape(
            outputs, shape=(outputs.shape[0],
                            outputs.shape[2]))

        y = self.F(outputs_reshaped)

        return y, s

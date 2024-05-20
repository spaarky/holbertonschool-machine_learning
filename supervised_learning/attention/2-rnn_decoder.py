#!/usr/bin/env python3
"""Summary"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Summary"""
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
        context_vector, _ = SelfAttention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, s = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.F(output)

        return x, s

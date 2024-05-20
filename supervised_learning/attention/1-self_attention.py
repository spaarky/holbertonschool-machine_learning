#!/usr/bin/env python3
"""Summary"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Summary"""
    def __init__(self, units):
        """Summary"""
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Summary"""
        s_expanded = tf.expand_dims(s_prev, 1)
        inputs = self.U(s_expanded)
        hidden = self.W(hidden_states)
        score = self.V(tf.nn.tanh(inputs + hidden))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * s_expanded
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

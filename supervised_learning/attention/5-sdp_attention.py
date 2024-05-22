#!/usr/bin/env python3
"""Summary"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Summary
    Q: tensor (..., seq_len_q, dk) containing the query matrix
    K: tensor (..., seq_len_v, dk) containing the key matrix
    V: tensor (..., seq_len_v, dv) containing the value matrix
    mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
    """

    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    keys_dim = tf.cast(tf.shape(K)[-1], tf.float32)

    scaled_attention_logits = matmul_qk / tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights

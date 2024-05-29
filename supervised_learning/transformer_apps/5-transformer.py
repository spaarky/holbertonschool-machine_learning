#!/usr/bin/env python3
"""Summary"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.compat.v1.enable_eager_execution()


class Dataset:
    """Summary"""
    def __init__(self, batch_size, max_len):
        """Class constructor"""

        def filter_max_len(x, y, max_length=max_len):
            """Summary"""
            return tf.logical_and(tf.size(x) <= max_len,
                                  tf.size(y) <= max_len)

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        PT, EN = self.tokenize_dataset(self.data_train)
        self.tokeniser_pt = PT
        self.tokeniser_en = EN

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(filter_max_len)
        self.data_train = self.data_train.cache()

        shu = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(shu)
        pad_shape = ([None], [None])
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)
        aux = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(aux)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)

    def tokenize_dataset(self, data):
        """Summary"""
        tokeniser_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15)

        tokeniser_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15)

        return tokeniser_pt, tokeniser_en

    def encode(self, pt, en):
        """Summary"""
        lang1 = [self.tokeniser_pt.vocab_size] + self.tokeniser_pt.encode(
            pt.numpy()) + [self.tokeniser_pt.vocab_size + 1]

        lang2 = [self.tokeniser_en.vocab_size] + self.tokeniser_en.encode(
            en.numpy()) + [self.tokeniser_en.vocab_size + 1]

        return lang1, lang2

    def tf_encode(self, pt, en):
        """Summary"""
        result_pt, result_en = tf.py_function(self.encode, [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

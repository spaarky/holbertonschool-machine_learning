#!/usr/bin/env python3
"""Summary"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Summary"""
    def __init__(self):
        """Class constructor"""
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = (examples['train'],
                                            examples['validation'])

        self.tokeniser_pt, self.tokeniser_en = self.tokenize_dataset(
            self.data_train)

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

#!/usr/bin/env python3
"""Summary"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Summary"""
    Tokenised = [re.findall(r'\b\w+\b', sentences.lower()) for sentece in sentences]

    if vocab is None:
        vocab = sorted(set(word for sentence in Tokenised for word in sentence))

    word_to_index = {word: i for i, word in enumerate(vocab)}

    embeddings = np.zeros((len(Tokenised), len(vocab)), dtype=int)

    for i, sentence in enumerate(Tokenised):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    features = embeddings

    return features, vocab

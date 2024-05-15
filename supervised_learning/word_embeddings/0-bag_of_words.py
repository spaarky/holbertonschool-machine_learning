#!/usr/bin/env python3
""" Creates a bag of words embedding matrix """
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """Summary"""
    tokenized = [re.findall(r'\b\w+\b', sentence.lower()) for sentence in sentences]

    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized for word in sentence))

    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, sentence in enumerate(tokenized):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    features = vocab

    return embeddings, features

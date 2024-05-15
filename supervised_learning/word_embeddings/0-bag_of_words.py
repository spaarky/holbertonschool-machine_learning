#!/usr/bin/env python3
""" Creates a bag of words embedding matrix """
import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    """Summary"""
    if not isinstance(sentences, list):
        raise TypeError("sentences should be a list.")

    preprocessed = []
    for sentence in sentences:
        sentences = re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower())
        preprocessed.append(sentences)

    list_words = []
    for sentence in preprocessed:
        words = re.findall(r'\w+', sentence)
        list_words.extend(words)

    if vocab is None:
        vocab = sorted(set(list_words))

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    features = vocab

    for i, sentence in enumerate(sentences):
        words = re.findall(r'\w+', sentence.lower())
        for word in words:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, features

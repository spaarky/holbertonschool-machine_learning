#!/usr/bin/env python3
"""
Creates a bag of words embedding matrix
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Summary"""
    vector = CountVectorizer(vocabulary=vocab)
    X = vector.fit_transform(sentences)
    features = vector.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features

#!/usr/bin/env python3
""" Summary """
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """Summary"""
    if vocab is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()

    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)

    embeddings = X.toarray()
    features = vocab

    return embeddings, features

#!/usr/bon/env python3
"""Summary"""
from gensim.models import Word2Vec
from keras.layers import Embedding


def gensim_to_keras(model):
    """Summary"""
    return model.wv.get_keras_embedding(train_embeddings=True)

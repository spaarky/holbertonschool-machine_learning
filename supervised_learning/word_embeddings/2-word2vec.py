#!/usr/bin/env python3
""" Summary """
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Summary"""
    sg = 0 if cbow else 1

    model = Word2Vec(sentences=sentences,
                     size=size,
                     window=window,
                     min_count=min_count,
                     workers=workers,
                     sg=sg, seed=seed,
                     negative=negative,
                     iter=iterations)

    model.save("word2vec.model")

    model = Word2Vec.load("word2vec.model")

    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.iter)

    return model

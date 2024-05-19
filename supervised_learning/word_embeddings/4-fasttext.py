#!/usr/bin/env python3
"""Summary"""
from gensim.models import FastText
from gensim.test.utils import datapath
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """Summary"""

    sg = 0 if cbow else 1
    model = FastText(sentences, size=size, min_count=min_count,
                     window=window, negative=negative, sg=sg,
                     seed=seed, workers=workers)

    model.save("fasttext.model")

    model = FastText.load("fasttext.model")

    # Train the model
    model.train(sentences=sentences,
                total_examples=model.corpus_count,
                epochs=iterations)

    return model

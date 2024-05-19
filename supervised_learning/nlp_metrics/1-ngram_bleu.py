#!/usr/bin/env python3
"""Summary"""
import numpy as np
from collections import Counter


def generate_ngrams(sentence, n):
    """Summary"""
    slices = []
    for i in range(n):
        slices.append(sentence[i:])
    return Counter(zip(*slices))


def generate_ref_ngrams(references, n):
    """Summary"""
    reference_ngrams = []
    for ref in references:
        reference_ngrams.append(generate_ngrams(ref, n))
    return reference_ngrams


def calculate_clipped_counts(sentence_ngrams, reference_ngrams):
    """Summary"""
    clipped_counts = {}
    for ngram, count in sentence_ngrams.items():
        max_ref_count = 0
        for ref_ngram in reference_ngrams:
            if ngram in ref_ngram:
                max_ref_count = max(max_ref_count, ref_ngram[ngram])
        clipped_counts[ngram] = min(count, max_ref_count)
    return clipped_counts


def calculate_precision(clipped_counts, sentence_ngrams):
    """Summary"""
    return sum(clipped_counts.values()) / max(1, sum(sentence_ngrams.values()))


def calculate_brevity_penalty(references, sentence):
    """Summary"""
    reference_lengths = [len(ref) for ref in references]
    len_sentence = len(sentence)
    closest_ref_length = min(
        reference_lengths,
        key=lambda ref_len: (abs(ref_len - len_sentence), ref_len))
    if len_sentence < closest_ref_length:
        return np.exp(1 - closest_ref_length / len_sentence)
    return 1


def ngram_bleu(references, sentence, n):
    """
    Summary
    """
    sentence_ngrams = generate_ngrams(sentence, n)
    reference_ngrams = generate_ref_ngrams(references, n)
    clipped_counts = calculate_clipped_counts(
        sentence_ngrams, reference_ngrams)
    precision = calculate_precision(clipped_counts, sentence_ngrams)
    brevity_penalty = calculate_brevity_penalty(references, sentence)
    return brevity_penalty * precision

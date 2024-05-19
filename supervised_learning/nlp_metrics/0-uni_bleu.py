#!/usr/bin/env python3
"""Summary"""
import numpy as np


def uni_bleu(references, sentence):
    """Summary"""
    # Calculate lenght of sentence
    sentence_len = len(sentence)

    # Find reference with closest length to sentence
    ref_len = []
    for ref in references:
        ref_len.append(len(ref))
    ref_len = np.array(ref_len)
    closest_ref_idx = np.argmin(np.abs(ref_len - sentence_len))
    closest_ref_len = len(references[closest_ref_idx])

    # Calculate unigram precision
    word_counts = {}
    for word in sentence:
        for ref in references:
            if word in ref:
                if word not in word_counts:
                    word_counts[word] = ref.count(word)
                else:
                    word_counts[word] = max(word_counts[word], ref.count(word))

    precision = sum(word_counts.values()) / sentence_len

    # Calculate brevity penalty
    if sentence_len > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    # Calculate BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score

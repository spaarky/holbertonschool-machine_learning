#!/usr/bin/env python3
"""Summary
"""
import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (pos. integer): the number of clusters

    Returns:
        pi (numpy.ndarray): shape (k,) containing the cluster priors
        m (numpy.ndarray): shape (k, d) containing the centroid means
        S (numpy.ndarray): shape (k, d, d) containing the covariance matrices
        clss (numpy.ndarray): shape (n,) containing the cluster indices for
            each data point
        bic (numpy.ndarray): shape (kmax - kmin + 1) containing the BIC value
            for each cluster size tested
    """

    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic

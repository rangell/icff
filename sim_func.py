import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def dot_prod(a, b):
    return a @ b.T


def jaccard_sim(a, b):
    """ Compute the pairwise jaccard similarity of sets of sets a and b. """
    intersect_size = np.sum((_a != 0) & (_b != 0) & (_a == _b), axis=-1)
    union_size = np.sum(np.abs(_a) + np.abs(_b), axis=-1) - intersect_size
    return intersect_size / union_size


def cos_sim(a, b):
    return cosine_similarity(a, b)

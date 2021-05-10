import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from IPython import embed


def dot_prod(a, b):
    return a @ b.T


def jaccard_sim(a, b):
    """ Compute the pairwise jaccard similarity of sets of sets a and b. """
    inf_mask = (np.any(a == -np.inf, axis=1)[:,None]
                | np.any(b == -np.inf, axis=1)[None, :])
    a[a == -np.inf] = 0
    b[b == -np.inf] = 0
    _a = a[:,None,:]
    _b = b[None, :, :]
    intersect_size = np.sum((_a != 0) & (_b != 0) & (_a == _b), axis=-1)
    union_size = np.sum(np.abs(_a) + np.abs(_b), axis=-1) - intersect_size

    scores = intersect_size / union_size
    scores[inf_mask] = -np.inf
    return scores


def cos_sim(a, b):
    #inf_mask = (np.any(a == -np.inf, axis=1)[:,None]
    #            | np.any(b == -np.inf, axis=1)[None, :])
    #a[a == -np.inf] = 0
    #b[b == -np.inf] = 0
    scores = cosine_similarity(a, b)
    if np.sum(a == -np.inf) + np.sum(b == -np.inf) > 0:
        embed()
        exit()
    #scores[inf_mask] = -np.inf
    return scores

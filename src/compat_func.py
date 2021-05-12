import numpy as np

from IPython import embed


def raw_overlap(node, constraint, num_points):
    rep = node.raw_rep
    pair_product = rep.multiply(constraint)
    if np.all(pair_product.data > 0):
        score = np.sum(pair_product > 0) / np.sum(constraint > 0) 
        return score if score < 1 else num_points * score
    else:
        return -np.inf


def transformed_overlap(node, constraint):
    # FIXME: this won't work
    assert False
    rep = node.transformed_rep
    pair_product = rep * constraint
    if np.array_equal(rep, np.ones_like(rep) * -np.inf) \
            or not np.all(pair_product >= 0):
        return -np.inf
    else:
        score = np.sum(pair_product > 0) / np.sum(constraint != 0)
        return score if score < 1 else num_points * score

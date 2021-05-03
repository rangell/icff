import numpy as np


def raw_overlap(node, constraint):
    rep = node.raw_rep
    if np.all((rep * constraint) >= 0):
        return np.sum(constraint == rep) / np.sum(constraint > 0)
    else:
        return -np.inf


def transformed_overlap(node, constraint):
    rep = node.transformed_rep
    if np.array_equal(rep, np.ones_like(rep) * -np.inf) \
            or not np.all((rep * constraint) >= 0):
        return -np.inf
    else:
        return np.sum(constraint == rep) / np.sum(constraint != 0)

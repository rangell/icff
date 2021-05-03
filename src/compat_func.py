import numpy as np


def raw_overlap(node, constraint):
    rep = node.raw_rep
    if np.all((rep * constraint) >= 0):
        return np.exp(np.sum(constraint == rep) / np.sum(constraint > 0))
    else:
        return -np.inf


def transformed_overlap(node, constraint):
    rep = node.transformed_rep
    if np.all((rep * constraint) >= 0):
        return np.exp(np.sum(constraint == rep) / np.sum(constraint != 0))
    else:
        return -np.inf

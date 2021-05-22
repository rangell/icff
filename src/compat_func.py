import copy
import numpy as np

from sparse_dot_mkl import dot_product_mkl

from utils import MIN_FLOAT

from IPython import embed


def raw_overlap(raw_points_normd, stacked_constraints, num_points):

    pos_feats = raw_points_normd.astype(bool).astype(float)
    extreme_raw_points_normd = copy.deepcopy(pos_feats)
    extreme_raw_points_normd *= np.inf
    extreme_constraints = copy.deepcopy(stacked_constraints).astype(float)
    extreme_constraints.data *= np.inf
    compat_mx = dot_product_mkl(
        extreme_raw_points_normd, extreme_constraints.T, dense=True
    )
    incompat_mx = ((compat_mx == -np.inf) | np.isnan(compat_mx))
    pos_overlap_scores = dot_product_mkl(
        pos_feats, stacked_constraints.T, dense=True
    )
    pos_overlap_scores = np.array(pos_overlap_scores)
    pos_overlap_mask = (pos_overlap_scores > 0)
    pos_feat_totals = np.array(np.sum(stacked_constraints > 0, axis=1).T)
    overlap_scores = pos_overlap_scores / pos_feat_totals
    overlap_scores *= pos_overlap_mask
    #overlap_scores[overlap_scores == 1] = num_points
    overlap_scores[incompat_mx] = -np.inf
    return np.asarray(overlap_scores)


def transformed_overlap(node, constraint):
    # FIXME: this won't work
    assert False
    rep = node.transformed_rep
    pair_product = rep * constraint
    if np.array_equal(rep, np.ones_like(rep) * -np.inf) \
            or not np.all(pair_product >= 0):
        return MIN_FLOAT
    else:
        score = np.sum(pair_product > 0) / np.sum(constraint != 0)
        return score if score < 1 else num_points * score

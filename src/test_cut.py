import time
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from sparse_dot_mkl import dot_product_mkl

from icff import intra_subcluster_energy
from utils import get_nil_rep

from IPython import embed


def constraint_satisfaction(opt,
                            node,
                            compat_func,
                            constraints,
                            num_points,
                            num_constraints):

    if len(constraints) > 0:
        constraint_scores = compat_func(
            node.raw_rep,
            sp.vstack(constraints),
            num_points if opt.super_compat_score else 1
        )

        if opt.compat_agg == 'avg':
            constraint_scores /= num_constraints

        # note: we only care about constraint_scores > 0
        constraint_scores[constraint_scores < 0] = 0
        constraint_scores = csr_matrix(constraint_scores)
    else:
        constraint_scores = get_nil_rep(rep_dim=len(constraints))

    return constraint_scores


def value_node(opt,
               node,
               sim_func,
               compat_func,
               constraints,
               incompat_mx,
               cost_per_cluster,
               num_points,
               num_constraints):

    # compute raw materials
    intra_energy = intra_subcluster_energy(opt, node, sim_func, num_points)
    intra_energy -= cost_per_cluster
    constraint_scores = constraint_satisfaction(
        opt, node, compat_func, constraints, num_points, num_constraints
    )
    compatibility_mx = (~incompat_mx).astype(float)
    agg_constraint_score = np.mean(
        dot_product_mkl(constraint_scores, compatibility_mx)
    )
    return (intra_energy,
            agg_constraint_score, 
            [node],
            constraint_scores,
            compatibility_mx)


def memoize_subcluster(opt,
                       node,
                       sim_func,
                       compat_func,
                       constraints,
                       incompat_mx,
                       cost_per_cluster,
                       num_points,
                       num_constraints):

    node_map = value_node(
        opt,
        node,
        sim_func,
        compat_func,
        constraints,
        incompat_mx,
        cost_per_cluster,
        num_points,
        num_constraints
    )

    if len(node.children) > 0:
        child_maps = [memoize_subcluster(opt,
                                         c, 
                                         sim_func,
                                         compat_func,
                                         constraints,
                                         incompat_mx,
                                         cost_per_cluster,
                                         num_points,
                                         num_constraints)
                        for c in node.children]
        assert len(child_maps) == 2 # restrict to binary trees for now

        print('At node: {}'.format(node.uid))

        # resolve child maps
        resolved_raw_score = np.sum([m[0] for m in child_maps])

        stacked_constraint_scores = sp.vstack(
            (child_maps[0][3], child_maps[1][3])
        )
        resolved_constraint_scores = np.max(
            stacked_constraint_scores, axis=0
        ).tocsr()
        right_mask = np.argmax(stacked_constraint_scores, axis=0).astype(bool)
        left_mask = ~right_mask

        right_compat_mask = (right_mask.T & right_mask).astype(float)
        left_compat_mask = (left_mask.T & left_mask).astype(float)

        left_incompat_mx = csr_matrix(1 - child_maps[0][4])
        right_incompat_mx = csr_matrix(1 - child_maps[1][4])
        
        resolved_compatibility_mx = 1 - (
            left_incompat_mx.multiply(left_compat_mask)
            + right_incompat_mx.multiply(right_compat_mask)
        ).toarray()
        resolved_agg_constraint_score = np.mean(
            dot_product_mkl(
                resolved_constraint_scores, resolved_compatibility_mx
            )
        )

        child_score = resolved_raw_score + resolved_agg_constraint_score
        if child_score > (node_map[0] + node_map[1]):
            resolved_cut_nodes = child_maps[0][2] + child_maps[1][2]
            node_map= (resolved_raw_score,
                       resolved_agg_constraint_score, 
                       resolved_cut_nodes,
                       resolved_constraint_scores,
                       resolved_compatibility_mx)

    return node_map


if __name__ == '__main__':

    print('Loading data...')
    with open('cut_test_input.pkl', 'rb') as f:
        in_data = pickle.load(f)
        (opt,
         pred_tree_nodes,
         sim_func,
         compat_func,
         constraints,
         incompat_mx,
         cost_per_cluster,
         num_points,
         num_constraints) = in_data
    print('Done.')

    time1 = time.time()
    root_value_map = memoize_subcluster(
        opt,
        pred_tree_nodes[-1],
        sim_func,
        compat_func,
        constraints,
        incompat_mx,
        cost_per_cluster,
        num_points,
        num_constraints
    )
    time2 = time.time()

    print('Cut time: {}'.format(time2 - time1))

    embed()
    exit()

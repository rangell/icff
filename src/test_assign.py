import copy
import time
import pickle
from collections import deque 
from heapq import heapify, heappush, heappop

from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from sparse_dot_mkl import dot_product_mkl

from IPython import embed


if __name__ == '__main__':

    print('Loading test data...')
    with open('assign_test_input.pkl', 'rb') as f:
        pred_tree_nodes, constraints, viable_placements = pickle.load(f)
    print('Done.')


    print('Reformating viable placements...')
    viable_placements = [[(s, n.uid) for s, n in placements]
                                for placements in viable_placements]
    print('Done.')

    print('Building tree incompatibility matrix...')
    num_tree_nodes = len(pred_tree_nodes)
    tree_incompat_mx = np.zeros((num_tree_nodes, num_tree_nodes), dtype=float)
    for node in pred_tree_nodes:
        if node.parent is None:
            tree_incompat_mx[node.uid][node.uid] = 1.0
            assert node == pred_tree_nodes[-1]
            break
        a, b = node.uid, node.parent.uid
        tree_incompat_mx[a][a] = 1.0
        tree_incompat_mx[a][b] = 1.0

    prev_tree_incompat_mx = np.zeros_like(tree_incompat_mx)
    curr_tree_incompat_mx = tree_incompat_mx
    while not np.array_equal(prev_tree_incompat_mx, curr_tree_incompat_mx):
        prev_tree_incompat_mx = copy.deepcopy(curr_tree_incompat_mx)
        curr_tree_incompat_mx = (
                (curr_tree_incompat_mx @ curr_tree_incompat_mx)
                + prev_tree_incompat_mx
        ).astype(bool).astype(float)

    curr_tree_incompat_mx = curr_tree_incompat_mx + curr_tree_incompat_mx.T
    tree_incompat_mx = curr_tree_incompat_mx.astype(bool)
    tree_incompat_mx = csr_matrix(tree_incompat_mx)
    print('Done.')

    print('Building constraint incompatibility matrix...')
    # organize constraints nicely
    Xi = sp.vstack(constraints)
    # compute incompatibility matrix
    extreme_constraints = copy.deepcopy(Xi)
    extreme_constraints.data *= np.inf
    constraint_incompat_mx = dot_product_mkl(
        extreme_constraints, extreme_constraints.T, dense=True
    )
    constraint_incompat_mx = ((constraint_incompat_mx == -np.inf)
                                | np.isnan(constraint_incompat_mx))
    constraint_incompat_mx = csr_matrix(constraint_incompat_mx)
    print('Done.')

    print('Greedy solution start...')

    # Setup:
    # - queue for every constraint (based on sorted list of viable placements)
    # - heap of proposed assignments for unassigned constraints
    # - list(?) of chosen constraints and assignments
    num_constraints = len(viable_placements)

    running_vp = [deque(l) for l in viable_placements]

    # note this is a min-heap so we negate the score
    to_pick_heap = [(i, *d.popleft()) for i, d in enumerate(running_vp)]
    to_pick_heap = [(-s, (c, t)) for c, s, t in to_pick_heap]
    heapify(to_pick_heap)

    picked_cidxs = np.array([])
    picked_tidxs = np.array([])

    while len(picked_cidxs) < num_constraints:
        _, (c, t) = heappop(to_pick_heap)

        if len(picked_cidxs) == 0:
            picked_cidxs = np.append(picked_cidxs, int(c))
            picked_tidxs = np.append(picked_tidxs, int(t))
            continue

        c_incompat_mask = constraint_incompat_mx[
            c, picked_cidxs
        ].toarray().reshape(-1,)

        t_incompat_mask = tree_incompat_mx[
            t, picked_tidxs[c_incompat_mask]
        ].toarray().reshape(-1,)

        if np.sum(t_incompat_mask) > 0:
            # if incompatible do the following
            _s, _t = running_vp[c].popleft()
            c_next_best = (-_s, (c, _t))
            heappush(to_pick_heap, c_next_best)
        else:
            print(len(picked_tidxs))
            picked_cidxs = np.append(picked_cidxs, int(c))
            picked_tidxs = np.append(picked_tidxs, int(t))

    embed()
    exit()


    # old implementation

    num_constraints = viable_placement_scores.shape[0]
    indices = np.array(np.argmax(viable_placement_scores, axis=1)).reshape(-1,)
    frozen_mask = np.zeros_like(indices).astype(bool)

    while np.sum(frozen_mask) < frozen_mask.size:
        proposed_scores = viable_placement_scores[np.arange(num_constraints), indices]
        proposed_scores = np.array(proposed_scores).reshape(-1,)
        best_unfrozen = np.arange(num_constraints)[~frozen_mask][np.argmax(proposed_scores[~frozen_mask])]

        incompat = False
        if np.sum(frozen_mask) != 0:
            frozen_indices = np.where(frozen_mask)[0]
            sub_constr_incompat = constraint_incompat_mx[best_unfrozen, frozen_indices]
            if sub_constr_incompat.size > 0:
                for _, frozen_idx in zip(*sub_constr_incompat.nonzero()):
                    if tree_incompat_mx[indices[best_unfrozen], indices[frozen_indices[frozen_idx]]]:
                        incompat = True
                        break
        if not incompat:
            frozen_mask[best_unfrozen] = True
            print(np.sum(frozen_mask))
        else:
            viable_placement_scores[best_unfrozen, indices[best_unfrozen]] = -np.inf
            indices = np.array(np.argmax(viable_placement_scores, axis=1)).reshape(-1,)

    embed()
    exit()

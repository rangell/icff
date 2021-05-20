import copy
import time
import pickle
import logging
from collections import deque, defaultdict
from heapq import heapify, heappush, heappop
from tqdm import tqdm 

from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from sparse_dot_mkl import dot_product_mkl
from ortools.sat.python import cp_model


from IPython import embed


logger = logging.getLogger(__name__)

def greedy_assign(pred_tree_nodes, constraints, viable_placements):
    logger.debug('Reformating viable placements')
    viable_placements = [[(s, n.uid) for s, n in placements]
                                for placements in viable_placements]

    # compute the tree incompatibility matrix
    logger.debug('Computing tree incompatibility matrix')
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

    # compute the transitive closure of the initial matrix
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

    # compute constraint incompatibility matrix
    logger.debug('Computing constraint incompatibility matrix')
    Xi = sp.vstack(constraints)
    extreme_constraints = copy.deepcopy(Xi)
    extreme_constraints.data *= np.inf
    constraint_incompat_mx = dot_product_mkl(
        extreme_constraints, extreme_constraints.T, dense=True
    )
    constraint_incompat_mx = ((constraint_incompat_mx == -np.inf)
                                | np.isnan(constraint_incompat_mx))
    constraint_incompat_mx = csr_matrix(constraint_incompat_mx)

    logger.debug('Begin greedy assignment of constraints to tree nodes')

    # Setup:
    # - queue for every constraint (based on sorted list of viable placements)
    # - heap of proposed assignments for unassigned constraints
    # - list(?) of chosen constraints and assignments

    max_assign_nuid = num_tree_nodes
    num_constraints = len(viable_placements)

    while True: # this is the "push-down" loop
 
        # running set of viable placements
        running_vp = [deque(l) for l in viable_placements]

        # note this is a min-heap so we negate the score
        to_pick_heap = [(i, *d.popleft()) for i, d in enumerate(running_vp)]
        to_pick_heap = [(-s, (c, t)) for c, s, t in to_pick_heap]
        heapify(to_pick_heap)

        picked_cidxs = np.array([], dtype=int)
        picked_tidxs = np.array([], dtype=int)

        while len(picked_cidxs) < num_constraints:
            _, (c, t) = heappop(to_pick_heap)

            if t > max_assign_nuid: # "push-down" condition
                # if incompatible do the following
                try:
                    _s, _t = running_vp[c].popleft()
                except:
                    max_assign_nuid = np.max(picked_tidxs) - 1
                    break
                c_next_best = (-_s, (c, _t))
                heappush(to_pick_heap, c_next_best)
                continue

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
                try:
                    _s, _t = running_vp[c].popleft()
                except:
                    max_assign_nuid = np.max(picked_tidxs) - 1
                    break
                c_next_best = (-_s, (c, _t))
                heappush(to_pick_heap, c_next_best)
            else:
                picked_cidxs = np.append(picked_cidxs, int(c))
                picked_tidxs = np.append(picked_tidxs, int(t))

        assert len(picked_cidxs) <= num_constraints
        if len(picked_cidxs) == num_constraints:
            break

    placements_out = defaultdict(set)
    for cuid, tuid in zip(picked_cidxs, picked_tidxs):
        placements_out[tuid].add(cuid)
    
    return placements_out

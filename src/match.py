import copy
from collections import defaultdict
from heapq import heappop, heappush
import numpy as np

from IPython import embed


class Frontier(object):
    def __init__(self, affinity, indices):
        self.ub_affinity = affinity
        self.indices = indices

    def __lt__(self, other):
        # used for min heap, so flip it
        return self.ub_affinity > other.ub_affinity 

    def __repr__(self):
        return '<ub_affinitiy: {} - indices: {}>'.format(
            self.ub_affinity,
            self.indices
        )


def uniqheappush(heap, inheap, frontier):
    if tuple(frontier.indices) in inheap:
        return False
    heappush(heap, frontier)
    inheap.add(tuple(frontier.indices))
    return True


def uniqheappop(heap, inheap):
    frontier = heappop(heap)
    inheap.discard(tuple(frontier.indices))
    return frontier


def match_constraints(constraints,
                      viable_assigns,
                      compat_fn,
                      allow_no_match=False):

    # organize constraints nicely
    Xi = np.vstack(constraints)

    # compute incompatibility matrix
    A = np.triu(np.any((Xi[:, None, :] * Xi[None, :, :]) < 0, axis=-1), k=1)

    # find best assignment using branch and bound (pruning) solution
    soln_frontier = []
    in_frontier = set()
    _ = uniqheappush(
        soln_frontier,
        in_frontier,
        Frontier(np.inf, [0] * len(viable_assigns))
    )
    compatible = False
    while len(soln_frontier) > 0:
        indices = uniqheappop(soln_frontier, in_frontier).indices

        invalid_prop = False
        prop_assigns = []
        for i, vp in enumerate(viable_assigns):
            try:
                prop_assigns.append(vp[indices[i]])
            except:
                if not allow_no_match:
                    invalid_prop = True
                    break
                if indices[i] == -1 or indices[i] >= len(viable_assigns[i]):
                    prop_assigns.append((0.0, None))

        if invalid_prop:
            continue

        # check if any incompatible pairs of constraints violate `compat_fn`
        compat_check_iter = zip(*np.where(A > 0))
        incompat_pairs = []
        for i, j in compat_check_iter:
            compatible = compat_fn(
                constraints[i],
                prop_assigns[i][1],
                constraints[j],
                prop_assigns[j][1]
            )
            if not compatible:
                incompat_pairs.append((i, j))
        
        if len(incompat_pairs) == 0:
            compatible = True
            break

        # current proposal must be incompatible, push available frontiers
        incompat_dict = defaultdict(list)
        for i, j in incompat_pairs:
            incompat_dict[i].append(j)

        # add frontiers
        for i, local_incompat in incompat_dict.items():
            # keep i
            keep_ub_affinity = sum(
                [x[0] for idx, x in enumerate(prop_assigns)
                    if idx not in local_incompat]
            )
            new_frontier_indices = copy.deepcopy(indices)
            can_keep_i = True
            for j in local_incompat:
                compatible = False
                _iter = enumerate(viable_assigns[j][indices[j]+1:])
                for offset, hypo_assign in _iter:
                    compatible = compat_fn(
                        constraints[i],
                        prop_assigns[i][1],
                        constraints[j],
                        hypo_assign[1]
                    )
                    if compatible:
                        new_frontier_indices[j] = indices[j] + 1 + offset
                        keep_ub_affinity += hypo_assign[0]
                        break

                if not compatible:
                    if not allow_no_match:
                        # can't keep i
                        can_keep_i = False
                        break
                    else:
                        # propose not to satisfy the jth constraint
                        new_frontier_indices[j] = -1
                        #keep_ub_affinity += 0.0  # vacuous, but for verbosity

            if can_keep_i:
                new_frontier = Frontier(
                    keep_ub_affinity, new_frontier_indices
                )
                uniqheappush(soln_frontier, in_frontier, new_frontier)

            # remove i
            if indices[i] + 1 < len(viable_assigns[i]) or allow_no_match:
                # create indices
                new_frontier_indices = copy.deepcopy(indices)

                # base remove ub affinity
                remove_ub_affinity = sum([x[0] for x in prop_assigns])
                remove_ub_affinity -= prop_assigns[i][0]

                if indices[i] + 1 < len(viable_assigns[i]):
                    # update indices
                    new_frontier_indices[i] = indices[i] + 1

                    # update remove ub affinitiy
                    new_i_affinity = viable_assigns[i][indices[i]+1][0]
                    remove_ub_affinity += new_i_affinity
                elif allow_no_match:
                    # update indices
                    new_frontier_indices[i] = -1

                    ## update remove ub affinitiy
                    #remove_ub_affinity += 0.0  # vacuous, but for verbosity

                new_frontier = Frontier(
                    remove_ub_affinity, new_frontier_indices
                )
                uniqheappush(soln_frontier, in_frontier, new_frontier)

    if compatible:
        scores, assigns = zip(*prop_assigns)
    else:
        return None
    return scores, assigns


def lca_check(constraint1, node1, constraint2, node2):
    """ Checks to see if lca(node1, node2) is not in {node1, node2}. """
    if node1 is None or node2 is None:
        return True

    nodes = sorted(
        [(node1.uid, node1), (node2.uid, node2)],
        key=lambda x: -x[0]
    )
    max_uid = nodes[0][0]

    while nodes[1][0] < max_uid:
        lower_node = nodes[1][1]
        par = lower_node.parent
        assert par is not None
        if par.uid == max_uid:
            return False
        nodes[1] = (par.uid, par)
        nodes = sorted(nodes, key=lambda x: -x[0])
        
    return True


def viable_match_check(constraint1, node1, constraint2, node2):
    if node1 is None or node2 is None:
        return True
    if node1 == node2 and np.any((constraint1 * constraint2) < 0):
        return False
    return True

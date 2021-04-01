from heapq import heappop, heappush
import numpy as np


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


def match_constraints(constraints, viable_assigns, compat_fn):
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
        
        # TODO: add "-1" possibility

        prop_assigns = [vp[indices[i]] 
            for i, vp in enumerate(viable_assigns)]

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
                        keep_ub_affinity += hypo_assign[0]
                        new_frontier_indices[j] = indices[j] + 1 + offset
                        break

                if not compatible:
                    # can't keep i
                    can_keep_i = False
                    break

            if can_keep_i:
                new_frontier = Frontier(
                    keep_ub_affinity, new_frontier_indices
                )
                uniqheappush(soln_frontier, in_frontier, new_frontier)

            # remove i
            if indices[i] + 1 < len(viable_assigns[i]):
                # create indices
                new_frontier_indices = copy.deepcopy(indices)
                new_frontier_indices[i] = indices[i] + 1

                # compute ub affinity
                new_i_affinity = viable_assigns[i][indices[i]+1][0]
                keep_ub_affinity = sum([x[0] for x in prop_assigns])
                keep_ub_affinity -= prop_assigns[i][0]
                keep_ub_affinity += new_i_affinity

                # build obj instance and add to heap
                new_frontier = Frontier(
                    keep_ub_affinity, new_frontier_indices
                )
                uniqheappush(soln_frontier, in_frontier, new_frontier)

    if compatible:
        _, assigns = zip(*prop_assigns)
    else:
        assigns = None
    return assigns


def lca_check(constraint1, node1, constraint2, node2):
    """ Checks to see if lca(node1, node2) is not in {node1, node2}. """
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
    if node1 == node2 and np.any((constraint1 * constraint2) < 0):
        return False
    return True

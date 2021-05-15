import numpy as np
import scipy.sparse as sp

from IPython import embed


def constraint_compatible_nodes(opt,
                                nodes,
                                constraints,
                                compat_func,
                                num_points):

    node_raw_reps = sp.vstack([n.raw_rep for n in nodes])
    overlap_scores = compat_func(
        node_raw_reps,
        sp.vstack(constraints),
        num_points if opt.super_compat_score else 1
    )

    viable_placements = []
    for scores in overlap_scores.T:
        node_indices = np.where(scores > 0)[0]
        compatible_nodes = [(scores[i], nodes[i]) for i in node_indices]
        viable_placements.append(
            sorted(
                compatible_nodes,
                key=lambda x: (x[0], x[1].uid),
                reverse=True
            )
        )

    return viable_placements


def lca(node1, node2):
    """ Returns least common ancestor of {node1, node2}. """
    nodes = sorted(
        [(node1.uid, node1), (node2.uid, node2)],
        key=lambda x: -x[0]
    )

    while nodes[0][0] != nodes[1][0]:
        lower_node = nodes[1][1]
        par = lower_node.parent
        assert par is not None
        nodes[1] = (par.uid, par)
        nodes = sorted(nodes, key=lambda x: -x[0])

    return nodes[0][1]

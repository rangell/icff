import numpy as np

from IPython import embed


def constraint_compatible_nodes(nodes, ff_constraint, sim_func):
    compatible_nodes = []
    for node in nodes:
        feat_alignment = node.transformed_rep * ff_constraint
        num_overlap = np.sum(feat_alignment > 0)
        num_violate = np.sum(feat_alignment < 0)
        if num_violate == 0 and num_overlap > 0:
            # NOTE: should this use the raw or transformed representation?
            #affinity = sim_func(node.raw_rep[None,:],
            #                    ff_constraint[None,:])[0][0]
            affinity = num_overlap / np.sum(ff_constraint != 0)
            compatible_nodes.append((affinity, node))
    return compatible_nodes


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

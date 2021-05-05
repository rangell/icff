import numpy as np

from IPython import embed


def constraint_compatible_nodes(opt,
                                nodes,
                                ff_constraint,
                                compat_func,
                                num_points):
    compatible_nodes = []
    for node in nodes:
        compat_score = compat_func(
            node, ff_constraint, num_points if opt.super_compat_score else 1
        )
        if compat_score > 0:
            compatible_nodes.append((compat_score, node))
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

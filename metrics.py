from collections import defaultdict
from itertools import combinations
from math import factorial

from tree_ops import lca

from IPython import embed


def dendrogram_purity(pred_tree_nodes, labels):
    gold_clusters = defaultdict(list)
    for node in pred_tree_nodes:
        if len(node.children) == 0: # node is a leaf
            gold_clusters[labels[node.uid]].append(node)

    purity = 0.0
    normalizer = 0.0
    for cluster in gold_clusters.values():
        m = len(cluster)
        normalizer += factorial(m) / (2 * factorial(m-2))
        for node1, node2 in combinations(cluster, 2):
            lca_leaves = lca(node1, node2).get_leaves()
            purity += sum([x in cluster for x in lca_leaves]) / len(lca_leaves)

    return purity / normalizer

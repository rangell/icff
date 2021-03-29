import os
import copy
import pickle
import random
from collections import defaultdict
from functools import reduce
from heapq import heappop, heappush, heapify

import higra as hg
from higra.higram import HorizontalCutExplorer
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import cosine_similarity

from IPython import embed


DATA_FNAME = 'synth_data.pkl'

# TODO:
# - add cmd line args
# - add wandb for larger experiments
# - 


class InferredTree(object):

    def __init__(self, nodes=[]):
        self.nodes = nodes

    def constraint_compatible_nodes(self, ff_constraint, sim_func):
        compatible_nodes = []
        for node in self.nodes:
            feat_alignment = node.rep * ff_constraint
            num_overlap = np.sum(feat_alignment > 0)
            num_violate = np.sum(feat_alignment < 0)
            if num_violate == 0 and num_overlap > 0:
                affinity = sim_func(node.rep[None,:], ff_constraint[None,:])[0][0]
                compatible_nodes.append((affinity, node))
        return compatible_nodes

    def lca_check(self, node1, node2):
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

    
class InferredNode(object):

    def __init__(self, uid, rep, children=[]):
        self.uid = uid
        self.rep = rep
        self.children = children
        self.parent = None

        for child in self.children:
            child.parent = self

    def get_leaves(self):
        if len(self.children) == 0:
            return [self]
        return reduce(
            lambda a, b : a + b,
            [c.get_leaves() for c in self.children]
        )


    def __repr__(self):
        return '<uid: {} - rep: {} - children: {} - parent: {}>'.format(
            self.uid,
            self.rep,
            [c.uid for c in self.children],
            self.parent.uid if self.parent is not None else 'None'
        )


############################ DATA GENERATION ##################################
def gen_data(num_entities, num_mentions, dim):
    entities, mentions, mention_labels = [], [], []
    block_size = dim // num_entities
    for ent_idx in range(num_entities):
        tmp_ent = np.zeros(dim, dtype=int)
        tmp_ent[ent_idx*block_size:(ent_idx+1)*block_size] = 1
        noise = (np.random.randint(0, 10, size=tmp_ent.shape) < 4).astype(int)
        tmp_ent |= noise
        entities.append(tmp_ent)
    entities = np.vstack(entities)

    # assert entities are non-nested!!!
    assert np.all(np.argmax(entities @ entities.T, axis=1)
                  == np.array(range(entities.shape[0])))

    for _ in range(num_mentions):
        ent_idx = random.randint(0, num_entities-1)
        mention_labels.append(ent_idx)
        while True:
            ent_mask = (np.random.randint(0, 10, size=tmp_ent.shape) < 4).astype(int)
            sample_mention = ent_mask & entities[ent_idx]
            subsets = np.all((sample_mention & entities) == sample_mention, axis=1)
            if subsets[ent_idx] and np.sum(subsets) == 1:
                mentions.append(sample_mention)
                break
    mentions = np.vstack(mentions)
    mention_labels = np.asarray(mention_labels)

    # HACK: entity reps are exactly the aggregation of all their mentions
    #        (no more, no less) -> this way we don't need to add attributes
    #        which are not present in the mentions
    for ent_idx in range(num_entities):
        mention_mask = (mention_labels == ent_idx)
        assert np.sum(mention_mask) > 0
        entities[ent_idx] = (np.sum(mentions[mention_mask], axis=0) > 0).astype(int)
        
    return entities, mentions, mention_labels


############################ SIM FUNCTIONS ###################################
def _dot_prod(a, b):
    return a @ b.T


def _jaccard_sim(a, b):
    """ Compute the pairwise jaccard similarity of sets of sets a and b. """
    intersect_size = np.sum((_a != 0) & (_b != 0) & (_a == _b), axis=-1)
    union_size = np.sum(np.abs(_a) + np.abs(_b), axis=-1) - intersect_size
    return intersect_size / union_size


def _cos_sim(a, b):
    return cosine_similarity(a, b)


############################ CLUSTERING ###################################
def custom_hac(points, sim_func):
    level_set = points.astype(float)
    uids = np.arange(level_set.shape[0])
    num_leaves = np.ones_like(uids)
    Z = []

    while level_set.shape[0] > 1:
        # compute dist matrix
        dist_mx = np.triu(1 - sim_func(level_set, level_set) + 1e-8, k=1)

        # this checks to see if two sub-clusters have features which violate
        # each other -- i.e. one has a feature and the other has *not* that
        # feature
        violate_mask = np.sum(
            np.abs(level_set[:,None,:] - level_set[None, :, :]) > 2,
            axis=-1
        ).astype(bool)

        # set violating indices to inf
        dist_mx[violate_mask] = np.inf

        # get next agglomeration
        agglom_coord = np.where(dist_mx == dist_mx[dist_mx != 0].min())
        agglom_coord = tuple(map(lambda x : x[0:1], agglom_coord))
        agglom_ind = np.array(list(map(lambda x : x[0], agglom_coord)))
        agglom_mask = np.zeros_like(uids, dtype=bool)
        agglom_mask[agglom_ind] = True
        if np.any(violate_mask[agglom_coord]):
            agglom_rep = np.ones_like(level_set[0]) * -np.inf
        else:
            agglom_rep = reduce(
                lambda a, b : a | b,
                level_set[agglom_mask].astype(int)
            )

        # update data structures
        not_agglom_mask = ~agglom_mask
        agglom_num_leaves = sum([num_leaves[x] for x in agglom_ind])
        Z.append(
            np.array(
                [float(uids[agglom_ind[0]]),
                 float(uids[agglom_ind[1]]),
                 float(dist_mx[agglom_coord]),
                 float(agglom_num_leaves)]
            )
        )
        level_set = np.concatenate(
            (level_set[not_agglom_mask], agglom_rep[None,:])
        )
        uids = np.concatenate(
            (uids[not_agglom_mask], np.array([np.max(uids) + 1]))
        )
        num_leaves = np.concatenate(
            (num_leaves[not_agglom_mask], np.array([agglom_num_leaves]))
        )

    # return the linkage matrix
    Z = np.vstack(Z)
    return Z


def cluster_points(points, labels, sim_func):
    num_points = points.shape[0]

    # run clustering and produce the linkage matrix
    Z = custom_hac(points, sim_func)
    altitudes = Z[:,2]
    cut_idx = np.argmin(np.diff(altitudes)) + 1 # cut after this merge idx
    
    # compute the predicted tree
    inferred_nodes = [InferredNode(i, r) for i, r in enumerate(points)]
    new_node_id = num_points
    assert new_node_id == len(inferred_nodes)
    for merge_idx, merger in enumerate(Z):
        lchild, rchild = int(merger[0]), int(merger[1])
        inferred_nodes.append(
            InferredNode(
                new_node_id,
                inferred_nodes[lchild].rep | inferred_nodes[rchild].rep,
                children=[inferred_nodes[lchild], inferred_nodes[rchild]]
            )
        )
        new_node_id += 1

        # compute the pred labels at the cut
        if merge_idx == cut_idx:
            forest_roots = [x for x in inferred_nodes if x.parent is None]
            leaf2parent = {
                leaf.uid : root.uid 
                    for root in forest_roots for leaf in root.get_leaves()
            }
            pred_labels = [leaf2parent[i] for i in range(num_points)]

    pred_tree = InferredTree(nodes=inferred_nodes)

    # compute the predicted entities 
    pred_entities_map = defaultdict(lambda : np.zeros_like(points[0]))
    for i, idx in enumerate(pred_labels):
        pred_entities_map[idx] |= points[i]
    pred_entities = np.asarray(list(pred_entities_map.values()))

    return pred_entities, pred_tree


def place_constraints(pred_tree, constraints, viable_placements):
    # organize constraints nicely
    Xi = np.vstack(constraints)

    # compute incompatibility matrix
    A = np.triu(np.any((Xi[:, None, :] * Xi[None, :, :]) < 0, axis=-1), k=1)

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

    # find best placement using branch and bound (pruning) solution
    soln_frontier = []
    in_frontier = set()
    _ = uniqheappush(
        soln_frontier,
        in_frontier,
        Frontier(np.inf, [0] * len(viable_placements))
    )
    compatible = False
    while len(soln_frontier) > 0:
        print(len(soln_frontier))
        indices = uniqheappop(soln_frontier, in_frontier).indices
        prop_placements = [vp[indices[i]] 
            for i, vp in enumerate(viable_placements)]

        # check if any incompatible pairs of constraints violate lca rule
        compat_check_iter = zip(*np.where(A > 0))
        incompat_pairs = []
        for i, j in compat_check_iter:
            compatible = pred_tree.lca_check(
                prop_placements[i][1], prop_placements[j][1]
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
                [x[0] for idx, x in enumerate(prop_placements)
                    if idx not in local_incompat]
            )
            new_frontier_indices = copy.deepcopy(indices)
            can_keep_i = True
            for j in local_incompat:
                compatible = False
                _iter = enumerate(viable_placements[j][indices[j]+1:])
                for offset, hypo_place in _iter:
                    compatible = pred_tree.lca_check(
                        prop_placements[i][1], hypo_place[1]
                    )
                    if compatible:
                        keep_ub_affinity += hypo_place[0]
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
            if indices[i] + 1 < len(viable_placements[i]):
                # create indices
                new_frontier_indices = copy.deepcopy(indices)
                new_frontier_indices[i] = indices[i] + 1

                # compute ub affinity
                new_i_affinity = viable_placements[i][indices[i]+1][0]
                keep_ub_affinity = sum([x[0] for x in prop_placements])
                keep_ub_affinity -= prop_placements[i][0]
                keep_ub_affinity += new_i_affinity

                # build obj instance and add to heap
                new_frontier = Frontier(
                    keep_ub_affinity, new_frontier_indices
                )
                uniqheappush(soln_frontier, in_frontier, new_frontier)

    if compatible:
        _, placements = zip(*prop_placements)
    else:
        placements = None
    return placements


if __name__ == '__main__':

    # fix this with command line args later
    seed = 27
    random.seed(seed)
    np.random.seed(seed)

    # get or create the synthetic data
    if not os.path.exists(DATA_FNAME):
        with open(DATA_FNAME, 'wb') as f:
            gold_entities, mentions, mention_labels = gen_data(2, 10, 16)
            pickle.dump((gold_entities, mentions, mention_labels), f)
    else:
        with open(DATA_FNAME, 'rb') as f:
            gold_entities, mentions, mention_labels = pickle.load(f)

    # declare similarity function
    sim_func = _cos_sim

    # cluster the points
    pred_entities, pred_tree = cluster_points(
        mentions, mention_labels, sim_func
    )

    # TESTING: generate a bunch of valid constraints
    constraints, viable_placements = [], []
    for i in range(25):
        # randomly generate valid feedback in the form of there-exists constraints
        pred_ent_idx = random.randint(0, len(pred_entities)-1)
        ff_pred_ent = pred_entities[pred_ent_idx]
        feat_intersect = (ff_pred_ent & gold_entities) == ff_pred_ent
        is_ent_subsets = np.all(feat_intersect, axis=1)
        num_ent_subsets = np.sum(is_ent_subsets)
        assert num_ent_subsets < 2
        if num_ent_subsets == 0:
            # split required (note: there might be a better way to do this...)
            tgt_ent_idx = np.argmax(np.sum(feat_intersect, axis=1))
            tgt_gold_ent = gold_entities[tgt_ent_idx]
            neg_match_feats = (feat_intersect[tgt_ent_idx] == False)
            while True:
                ff_mask = np.random.randint(0, 2, size=super_gold_ent.size)
                if np.sum(neg_match_feats * ff_mask) > 0:
                    break
            ff_constraint = (2*tgt_gold_ent - 1) * ff_mask
        else:
            assert num_ent_subsets == 1
            # merge required
            super_gold_ent = gold_entities[np.argmax(is_ent_subsets)]
            ff_mask = np.random.randint(0, 2, size=super_gold_ent.size)
            ff_constraint = (2*super_gold_ent - 1) * ff_mask

        compatible_nodes = pred_tree.constraint_compatible_nodes(
            ff_constraint, sim_func
        )

        constraints.append(ff_constraint)
        viable_placements.append(sorted(compatible_nodes, key=lambda x: -x[0]))

    # solve structured prediction problem of jointly placing the constraints
    resolved_placements = place_constraints(
        pred_tree, constraints, viable_placements
    )
    assert resolved_placements is not None

    embed()
    exit()


    constraint_leaves = max_compatible_node.get_leaves()


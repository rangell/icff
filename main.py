import os
import copy
import pickle
import random
from collections import defaultdict
from heapq import heappop, heappush, heapify
from functools import reduce

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score as adj_rand
from sklearn.metrics import adjusted_mutual_info_score as adj_mi
from sklearn.metrics.pairwise import cosine_similarity

from metrics import dendrogram_purity
from tree_ops import (constraint_compatible_nodes,
                      lca_check)
from tree_node import TreeNode

from IPython import embed


DATA_FNAME = 'synth_data.pkl'

# TODO:
# - add cmd line args
# - add wandb for larger experiments
# - 


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


def cut_objective(cut_frontier, sim_func, constraints):
    # trying to maximize the value produced by this function

    def expected_sim(reps):
        if reps.shape[0] == 1:
            return 0.0
        sim_mx = np.triu(sim_func(reps, reps) + 1e-8, k=1)
        return np.mean(sim_mx[sim_mx > 0])

    # compute expected external cluster sim
    extern_cluster_reps = np.vstack([n.transformed_rep for n in cut_frontier])
    extern_cluster_sim = expected_sim(extern_cluster_reps)

    # compute expected internal cluster sim
    intern_cluster_sims = []
    for n in cut_frontier:
        intern_cluster_reps = np.vstack(
            [l.transformed_rep for l in n.get_leaves()]
        )
        intern_cluster_sims.append(expected_sim(intern_cluster_reps))
    intern_cluster_sim = np.mean(intern_cluster_sims)

    # before constraints
    obj_val = np.exp(intern_cluster_sim - extern_cluster_sim)

    # TODO: incorporate constraint violation + attr add penalty

    ## track the cuts we're evaluating
    #print(str(obj_val) + '\n' + str(cut_frontier))

    return obj_val


def get_opt_tree_cut(cut_frontier, sim_func, constraints):
    max_cut = cut_frontier
    max_cut_score = cut_objective(cut_frontier, sim_func, constraints)
    for i, node in enumerate(cut_frontier):
        if len(node.children) > 0:
            alter_cut_front = cut_frontier[:i]\
                              + node.children\
                              + cut_frontier[i+1:]
            alter_cut, alter_cut_score = get_opt_tree_cut(
                alter_cut_front, sim_func, constraints
            )
            if alter_cut_score > max_cut_score:
                max_cut = alter_cut
                max_cut_score = alter_cut_score

    return max_cut, max_cut_score


def cluster_points(leaf_nodes, labels, sim_func, constraints):
    # pull out all of the points
    points = np.vstack([x.transformed_rep for x in leaf_nodes])
    num_points = points.shape[0]

    # run clustering and produce the linkage matrix
    Z = custom_hac(points, sim_func)

    # build the tree
    pred_tree_nodes = copy.deepcopy(leaf_nodes)
    new_node_id = num_points
    assert new_node_id == len(pred_tree_nodes)
    for merge_idx, merger in enumerate(Z):
        lchild, rchild = int(merger[0]), int(merger[1])
        pred_tree_nodes.append(
            TreeNode(
                new_node_id,
                pred_tree_nodes[lchild].raw_rep | pred_tree_nodes[rchild].raw_rep,
                transformed_rep=(pred_tree_nodes[lchild].transformed_rep
                                | pred_tree_nodes[rchild].transformed_rep),
                children=[pred_tree_nodes[lchild], pred_tree_nodes[rchild]]
            )
        )
        new_node_id += 1

    # find the best cut
    cut_frontier_nodes, cut_obj_score = get_opt_tree_cut(
        pred_tree_nodes[-1].children, sim_func, constraints
    )

    # the predicted entities canonicalization
    pred_canon_ents = np.vstack(
        [n.transformed_rep for n in cut_frontier_nodes]
    )

    # produce the predicted labels for leaves
    pred_labels = np.zeros_like(labels)
    for i, n in enumerate(cut_frontier_nodes):
        for x in n.get_leaves():
            pred_labels[x.uid] = i
    

    # compute metrics
    dp = dendrogram_purity(pred_tree_nodes, labels)
    adj_rand_idx = adj_rand(pred_labels, labels)
    adj_mut_info = adj_mi(pred_labels, labels)

    metrics = {
        'dp' : dp,
        'adj_rand_idx' : adj_rand_idx,
        'adj_mut_info' : adj_mut_info,
        'cut_obj_score' : cut_obj_score
    }

    return pred_canon_ents, pred_labels, pred_tree_nodes, metrics


def place_constraints(constraints, viable_placements):
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
        indices = uniqheappop(soln_frontier, in_frontier).indices
        prop_placements = [vp[indices[i]] 
            for i, vp in enumerate(viable_placements)]

        # check if any incompatible pairs of constraints violate lca rule
        compat_check_iter = zip(*np.where(A > 0))
        incompat_pairs = []
        for i, j in compat_check_iter:
            compatible = lca_check(
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
                    compatible = lca_check(
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


def gen_constraint(gold_entities,
                   pred_canon_ents,
                   pred_tree_nodes,
                   sim_func,
                   num_to_generate=1):
    constraints = []
    for _ in range(num_to_generate):
        # randomly generate valid feedback in the form of there-exists constraints
        pred_ent_idx = random.randint(0, len(pred_canon_ents)-1)
        ff_pred_ent = pred_canon_ents[pred_ent_idx]
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
                ff_mask = np.random.randint(0, 2, size=tgt_gold_ent.size)
                if np.sum(neg_match_feats * ff_mask) > 0:
                    break
            ff_constraint = (2*tgt_gold_ent - 1) * ff_mask
        else:
            assert num_ent_subsets == 1
            # merge required
            super_gold_ent = gold_entities[np.argmax(is_ent_subsets)]
            ff_mask = np.random.randint(0, 2, size=super_gold_ent.size)
            ff_constraint = (2*super_gold_ent - 1) * ff_mask

        constraints.append(ff_constraint)

    return constraints


def run_dummy_icff(gold_entities,
                   mentions,
                   mention_labels,
                   sim_func,
                   rounds=1):
    constraints = []

    # construct tree node objects for leaves
    leaves = [TreeNode(i, m_rep) for i, m_rep in enumerate(mentions)]

    for _ in range(rounds):
        # cluster the points
        out = cluster_points(
            leaves, mention_labels, sim_func, constraints
        )
        pred_canon_ents, pred_labels, pred_tree_nodes, metrics = out

        # generate constraints and viable places given predictions
        new_constraints = gen_constraint(
            gold_entities, pred_canon_ents, pred_tree_nodes, sim_func
        )

        # update constraints and viable placements
        constraints.extend(new_constraints)
        viable_placements = []
        for xi in constraints:
            compatible_nodes = constraint_compatible_nodes(
                pred_tree_nodes, xi, sim_func
            )
            viable_placements.append(
                    sorted(compatible_nodes, key=lambda x: -x[0])
            )

        # solve structured prediction problem of jointly placing the constraints
        resolved_placements = place_constraints(constraints, viable_placements)
        assert resolved_placements is not None

        embed()
        exit()


        # TODO: project resolved constraint placements to leaves

    embed()
    exit()


def main():
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

    # run the core function
    run_dummy_icff(gold_entities, mentions, mention_labels, sim_func)


if __name__ == '__main__':
    main()

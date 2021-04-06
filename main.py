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

from data import gen_data
from match import match_constraints, lca_check, viable_match_check
from metrics import dendrogram_purity
from sim_func import dot_prod, jaccard_sim, cos_sim
from tree_ops import constraint_compatible_nodes
from tree_node import TreeNode

from IPython import embed


DATA_FNAME = 'synth_data.pkl'

# TODO:
# - add cmd line args
# - add wandb for larger experiments
# - 


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
    """ trying to maximize the value produced by this function. """

    # single cluster case
    if len(cut_frontier) < 2:
        return -np.inf

    def expected_sim(reps):
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
        if intern_cluster_reps.shape[0] == 1:
            continue
        intern_cluster_sims.append(expected_sim(intern_cluster_reps))

    # all singletons case
    if len(intern_cluster_sims) < 1:
        return -np.inf

    # o.w.
    intern_cluster_sim = np.mean(intern_cluster_sims)

    # before constraints
    log_obj_val = intern_cluster_sim - extern_cluster_sim

    # incorporate constraint violation + attr add penalty
    if len(constraints) > 0:
        # need to compute the viable matches between constraints and cut
        viable_assigns = []
        for xi in constraints:
            local_viable_assigns = []
            for sub_cluster in cut_frontier:
                if np.all((sub_cluster.raw_rep * xi) > 0):
                    assign_aff = (np.sum(xi == sub_cluster.raw_rep)
                                  / np.sum(xi > 0))
                    local_viable_assigns.append((assign_aff, sub_cluster))
            viable_assigns.append(local_viable_assigns)
        
        # we have a matching problem between constraints and cut
        match_out = match_constraints(
            constraints,
            viable_assigns,
            viable_match_check,
            allow_no_match=True
        )
        assert match_out is not None
        match_scores, _ = match_out

        # update log_obj_val
        constraint_satisfaction = np.mean(match_scores)
        log_obj_val += constraint_satisfaction

    obj_val = np.exp(log_obj_val)

    #if len(constraints) > 0:
    #    # track the cuts we're evaluating
    #    print(str(obj_val) + '\n' + str(cut_frontier))

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
    pred_tree_nodes = copy.copy(leaf_nodes) # shallow copy on purpose!
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
                   rounds=10):
    constraints = []

    # construct tree node objects for leaves
    leaves = [TreeNode(i, m_rep) for i, m_rep in enumerate(mentions)]

    for r in range(rounds):
        # cluster the points
        out = cluster_points(
            leaves, mention_labels, sim_func, constraints
        )
        pred_canon_ents, pred_labels, pred_tree_nodes, metrics = out

        print(metrics)

        # TODO: add logger to print metrics

        # TODO: add check to see if perfect clustering is returned

        # generate constraints and viable places given predictions
        new_constraints = gen_constraint(
            gold_entities,
            pred_canon_ents,
            pred_tree_nodes,
            sim_func,
            num_to_generate=3
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
        placements_out = match_constraints(
            constraints,
            viable_placements,
            lca_check,
            allow_no_match=False    # may change this arg later
        )
        assert placements_out is not None
        _, resolved_placements = placements_out

        # reset all leaf transformed_rep's
        for node in leaves:
            node.transformed_rep = copy.deepcopy(node.raw_rep)

        # project resolved constraint placements to leaves
        for xi, rp in zip(constraints, resolved_placements):
            for node in rp.get_leaves():
                node.transformed_rep |= xi

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

    # declare similarity function with function pointer
    sim_func = cos_sim

    # run the core function
    run_dummy_icff(gold_entities, mentions, mention_labels, sim_func)


if __name__ == '__main__':
    main()

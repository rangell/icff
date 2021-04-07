import os
import copy
import pickle
import random
import logging
import argparse
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
from utils import initialize_exp

from IPython import embed


logger = logging.getLogger(__name__)

# TODO:
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
        assert sim_mx[sim_mx != 0].size > 0
        return np.mean(sim_mx[sim_mx != 0])

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
    assert len(intern_cluster_sims) > 0
    intern_cluster_sim = np.mean(intern_cluster_sims)

    # before constraints
    obj_val = intern_cluster_sim - extern_cluster_sim

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

        # update obj_val
        assert len(match_scores) > 0
        constraint_satisfaction = np.mean(match_scores)
        obj_val += constraint_satisfaction

    return obj_val


def sample_single_cut(node):
    split_prob = 1.0 - (1.0 / node.num_cuts)
    #split_prob = 1.0 - (1.0 / (1.0 + np.log(node.num_cuts)))

    noise_sample = random.uniform(0, 1)
    if noise_sample < split_prob:
        # split down
        return reduce(
            lambda l1, l2 : l1 + l2,
            [sample_single_cut(c) for c in node.children]
        )
    # else, keep this node in cut frontier
    return [node]


def get_opt_tree_cut(pred_tree_nodes, leaf_nodes, sim_func, constraints):
    # simulated annealing approach

    max_steps = 100

    root_node = pred_tree_nodes[-1]
    curr_cut = sample_single_cut(root_node)
    curr_cut_score = cut_objective(
        curr_cut, sim_func, constraints
    )
    max_cut, max_cut_score = copy.copy(curr_cut), curr_cut_score
    root_node = pred_tree_nodes[-1]

    for step_num in range(max_steps):
        temp = 1.0 / np.sqrt((step_num + 1.0) / (max_steps + 1.0)) - 1.0  # might change this function later

        # sample random neighbor cut -> proposed cut
        prop_cut = copy.copy(curr_cut)
        sampled_node_idx = random.randint(0, len(curr_cut)-1)
        sampled_node = prop_cut[sampled_node_idx]
        split_flag = random.randint(0, 1) and (len(sampled_node.children) > 0)
        if sampled_node.parent is None or split_flag:
            # split
            del prop_cut[sampled_node_idx]
            prop_cut.extend(sampled_node.children)
        else:
            # merge
            sampled_node_parent = sampled_node.parent
            idxs_to_del = [i for i, n in enumerate(prop_cut)
                                if n.parent == sampled_node_parent]
            idxs_to_del = sorted(idxs_to_del, reverse=True) 
            for idx in idxs_to_del:
                del prop_cut[idx]
            prop_cut.append(sampled_node_parent)

        # compute objective for proposed cut
        prop_cut_score = cut_objective(
            prop_cut, sim_func, constraints
        )

        # update max_cut if proposed is better
        if prop_cut_score > max_cut_score:
            max_cut = copy.copy(prop_cut)
            max_cut_score = prop_cut_score

        # accept proposed cut or not
        accept_prob = np.min(
            (1.0, np.exp((prop_cut_score - curr_cut_score) / temp))
        )
        noise_sample = random.uniform(0, 1)
        if noise_sample < accept_prob:
            curr_cut = copy.copy(prop_cut)
            curr_cut_score = prop_cut_score

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
        pred_tree_nodes, leaf_nodes, sim_func, constraints
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


def run_mock_icff(opt,
                  gold_entities,
                  mentions,
                  mention_labels,
                  sim_func):
    constraints = []

    # construct tree node objects for leaves
    leaves = [TreeNode(i, m_rep) for i, m_rep in enumerate(mentions)]

    for r in range(opt.max_rounds):
        # cluster the points
        out = cluster_points(
            leaves, mention_labels, sim_func, constraints
        )
        pred_canon_ents, pred_labels, pred_tree_nodes, metrics = out

        logger.info("round: {} - metrics: {}".format(r, metrics))
        if metrics['adj_rand_idx'] == 1.0:
            logger.info("perfect clustering reached in {} rounds".format(r))
            break

        # generate constraints and viable places given predictions
        new_constraints = gen_constraint(
            gold_entities,
            pred_canon_ents,
            pred_tree_nodes,
            sim_func,
            num_to_generate=opt.num_constraints_per_round
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



def get_opt():

    # TODO: add conditional opts (e.g. diff opts for synthetic vs. real data)

    parser = argparse.ArgumentParser() 
    parser.add_argument('--seed', type=int, default=27,
                        help="random seed for initialization")
    parser.add_argument('--debug', action='store_true',
                        help="Enables and disables certain opts for debugging")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory for this run.")
    parser.add_argument("--data_dir", default=None, type=str,
                        help="The directory where data is stored.")
    parser.add_argument('--num_entities', type=int, default=2,
                        help="number of entities to generate when generating"\
                             "synthetic data")
    parser.add_argument('--num_mentions', type=int, default=10,
                        help="number of mentions to generate when generating"\
                             "synthetic data")
    parser.add_argument('--max_rounds', type=int, default=100,
                        help="number of rounds to generate feedback for")
    parser.add_argument('--num_constraints_per_round', type=int, default=1,
                        help="number of constraints to generate per round")
    parser.add_argument('--data_dim', type=int, default=16,
                        help="number of possible features (i.e. dimension of"\
                             "vector representation of points")

    opt = parser.parse_args()

    # check to make sure there are no issues with the specified opts
    check_opt(opt)

    return opt


def check_opt(opt):
    # TODO: do a bunch of checks on the options
    pass


def main():
    # get command line options
    opt = get_opt()

    # initialize the experiment
    initialize_exp(opt)

    # get or create the synthetic data
    data_fname = '{}/synth_data-{}_{}_{}-{}.pkl'.format(
        opt.data_dir,
        opt.num_entities,
        opt.num_mentions,
        opt.data_dim,
        opt.seed
    )
    if not os.path.exists(data_fname):
        with open(data_fname, 'wb') as f:
            gold_entities, mentions, mention_labels = gen_data(
                    opt.num_entities, opt.num_mentions, opt.data_dim
            )
            pickle.dump((gold_entities, mentions, mention_labels), f)
    else:
        with open(data_fname, 'rb') as f:
            gold_entities, mentions, mention_labels = pickle.load(f)

    # declare similarity function with function pointer
    sim_func = cos_sim

    # run the core function
    run_mock_icff(opt, gold_entities, mentions, mention_labels, sim_func)


if __name__ == '__main__':
    main()

import os
import gc
import copy
import time
import pickle
import random
import logging
import argparse
from collections import defaultdict
from heapq import heappop, heappush, heapify
from functools import reduce
from itertools import product
from tqdm import tqdm, trange

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score as adj_rand
from sklearn.metrics import adjusted_mutual_info_score as adj_mi
from sklearn.preprocessing import normalize

from sparse_dot_mkl import dot_product_mkl

import higra as hg

from compat_func import raw_overlap, transformed_overlap
from data import gen_data
from sim_func import dot_prod, jaccard_sim, cos_sim
from tree_ops import constraint_compatible_nodes
from tree_node import TreeNode
from utils import (MIN_FLOAT,
                   initialize_exp,
                   sparse_agglom_rep,
                   get_nil_rep,
                   get_tfidf_normd)

from IPython import embed


logger = logging.getLogger(__name__)


def dense_update(full, mask, new):
    return np.concatenate((full[mask], np.array([new])))


def custom_hac(opt, points, constraints):
    num_points = points.shape[0]
    level_set = points.astype(float)
    Z = []

    uids = np.arange(level_set.shape[0])
    num_leaves = np.ones_like(uids)

    level_set_normd = copy.deepcopy(level_set)
    points_normd = copy.deepcopy(level_set)
    sim_mx = dot_product_mkl(level_set_normd, level_set_normd.T, dense=True)

    set_union = list(range(points.shape[0]))
    must_link_gen = iter([(a, b) for p, a, b in constraints if p == np.inf])
    forced_mergers_left = True

    cannot_link_pairs = list(set([(a, b) for p, a, b in constraints if p == -np.inf]))
    cannot_link_pairs += [(b, a) for a, b in cannot_link_pairs]
    cannot_link_idxs = tuple(np.array(l) for l in zip(*cannot_link_pairs))
    if len(cannot_link_pairs) > 0:
        sim_mx[cannot_link_idxs] = MIN_FLOAT # don't choose these if we can avoid it

    # for computing best cut
    best_cut_score = MIN_FLOAT
    best_cut = copy.deepcopy(uids)
    cluster_ids = np.arange(num_points)
    intra_cluster_energies = np.ones_like(cluster_ids) * (1 / num_points)
 
    for _ in trange(level_set.shape[0] - 1):

        sim_mx[tuple([np.arange(level_set.shape[0])]*2)] = -np.inf # ignore diag

        # get next agglomeration
        if forced_mergers_left:
            try:
                while True:
                    luid, ruid = next(must_link_gen)
                    # set union check
                    while luid != set_union[luid]:
                        luid = set_union[luid]
                    while ruid != set_union[ruid]:
                        ruid = set_union[ruid]
                    luid = np.where(uids == luid)[0].item()
                    ruid = np.where(uids == ruid)[0].item()
                    if luid != ruid:
                        break
                agglom_ind = np.array([luid, ruid])
            except StopIteration:
                forced_mergers_left = False

        if not forced_mergers_left:
            agglom_coord = np.where(sim_mx == sim_mx.max())
            agglom_coord = tuple(map(lambda x : x[0:1], agglom_coord))
            agglom_ind = np.array(list(map(lambda x : x[0], agglom_coord)))

        agglom_mask = np.zeros_like(uids, dtype=bool)
        agglom_mask[agglom_ind] = True

        if forced_mergers_left or sim_mx[agglom_coord].item() > MIN_FLOAT:
            agglom_rep = sparse_agglom_rep(level_set[agglom_mask])
        else:
            agglom_rep = get_nil_rep(rep_dim=level_set.shape[1])

        assert np.sum(agglom_mask) == 2

        # update data structures
        linkage_score = np.inf if forced_mergers_left else sim_mx[agglom_coord]
        not_agglom_mask = ~agglom_mask
        agglom_num_leaves = sum([num_leaves[x] for x in agglom_ind])

        # update linkage matrix
        Z.append(
            np.array(
                [float(uids[agglom_ind[0]]),
                 float(uids[agglom_ind[1]]),
                 float(linkage_score),
                 float(agglom_num_leaves)]
            )
        )

        # update level set
        level_set = sp.vstack(
            (level_set[not_agglom_mask], agglom_rep)
        )

        # update sim_mx
        num_untouched = np.sum(not_agglom_mask)
        sim_mx = sim_mx[not_agglom_mask[:,None] & not_agglom_mask[None,:]]
        sim_mx = sim_mx.reshape(num_untouched, num_untouched)
        sim_mx = np.concatenate(
            (sim_mx, np.ones((1, num_untouched)) * -np.inf), axis=0
        )
        agglom_rep_normd = normalize(
            agglom_rep / agglom_num_leaves, norm='l2', axis=1
        )
        level_set_normd = sp.vstack(
            (level_set_normd[not_agglom_mask], agglom_rep_normd)
        )
        new_sims = dot_product_mkl(
            level_set_normd, agglom_rep_normd.T, dense=True
        )
        sim_mx = np.concatenate((sim_mx, new_sims), axis=1)

        # update set union
        next_uid = np.max(uids) + 1
        new_cluster_mask = np.isin(cluster_ids, uids[agglom_mask])
        cluster_ids[new_cluster_mask] = next_uid
        assert next_uid == len(set_union)
        set_union.append(next_uid)
        for agglom_idx in agglom_ind:
            set_union[uids[agglom_idx]] = next_uid

        # update uids list
        uids = np.concatenate(
            (uids[not_agglom_mask], np.array([next_uid]))
        )
        # update new_leaves list
        num_leaves = np.concatenate(
            (num_leaves[not_agglom_mask], np.array([agglom_num_leaves]))
        )

        # update cannot_link_idxs
        if len(cannot_link_pairs) > 0:
            def remap_cannot_idxs(idx):
                if idx in agglom_ind:
                    return uids.size - 1
                return idx - np.sum(idx > agglom_ind)
            v_remap_cannot_idxs = np.vectorize(remap_cannot_idxs)
            cannot_link_idxs = tuple(
                v_remap_cannot_idxs(l) for l in cannot_link_idxs
            )

            sim_mx[cannot_link_idxs] = MIN_FLOAT # don't choose these if we can avoid it

        # update intra cluster energies
        agglom_energy = np.sum(
            dot_product_mkl(
                points_normd[new_cluster_mask],
                agglom_rep_normd.T,
                dense=True
            )
        )
        agglom_energy /= num_points
        intra_cluster_energies = dense_update(
            intra_cluster_energies, not_agglom_mask, agglom_energy
        )

        cut_score = np.sum(intra_cluster_energies)\
                    - (opt.cost_per_cluster * intra_cluster_energies.size)

        if level_set.shape[0] == 1: # don't allow cut at 1 cluster
            continue

        if linkage_score <= MIN_FLOAT: # don't compute cut score of cannot-link
            continue

        if forced_mergers_left: # reset best while there are still forced_mergers_left (this works!)
            best_cut_score = MIN_FLOAT
            best_cut = copy.deepcopy(uids)

        if cut_score >= best_cut_score:
            best_cut_score = cut_score
            best_cut = copy.deepcopy(uids)

    # sanity check
    assert level_set.shape[0] == 1

    # return the linkage matrix
    Z = np.vstack(Z)

    return Z, best_cut, best_cut_score


def cluster_points(opt,
                   leaf_nodes,
                   labels,
                   sim_func,
                   constraints,
                   cost_per_cluster):
    # pull out all of the points
    points = sp.vstack([x.transformed_rep for x in leaf_nodes])
    num_points = points.shape[0]
    num_constraints = len(constraints)

    # run clustering and produce the linkage matrix
    Z, best_cut, cut_obj_score = custom_hac(opt, points, constraints)

    # build the tree
    pred_tree_nodes = copy.copy(leaf_nodes) # shallow copy on purpose!
    new_node_id = num_points
    struct_node_list = np.arange(2*num_points - 1) # used for higra's dp
    assert new_node_id == len(pred_tree_nodes)
    for merge_idx, merger in enumerate(Z):
        lchild, rchild, score = int(merger[0]), int(merger[1]), merger[2]

        struct_node_list[lchild] = new_node_id
        struct_node_list[rchild] = new_node_id

        lc_rr = pred_tree_nodes[lchild].raw_rep
        rc_rr = pred_tree_nodes[rchild].raw_rep
        lc_tr = pred_tree_nodes[lchild].transformed_rep
        rc_tr = pred_tree_nodes[rchild].transformed_rep

        agglom_raw_rep = sparse_agglom_rep(sp.vstack((lc_rr, rc_rr)))
        if score > MIN_FLOAT:
            agglom_transformed_rep = sparse_agglom_rep(
                sp.vstack((lc_tr, rc_tr))
            )
        else:
            agglom_transformed_rep = get_nil_rep(rep_dim=lc_rr.shape[1])

        pred_tree_nodes.append(
            TreeNode(
                new_node_id,
                agglom_raw_rep,
                transformed_rep=agglom_transformed_rep,
                children=[pred_tree_nodes[lchild], pred_tree_nodes[rchild]]
            )
        )
        new_node_id += 1

    # maximally pure mergers
    maximally_pure_mergers = [n for n in pred_tree_nodes
            if n.label is not None and n.parent.label is None]

    # extract cut frontier
    cut_frontier_nodes = [pred_tree_nodes[i] for i in best_cut]

    # the predicted entities canonicalization
    pred_canon_ents = sp.vstack(
        [n.raw_rep.astype(bool).astype(float) for n in cut_frontier_nodes]
    )

    # produce the predicted labels for leaves
    pred_labels = np.zeros_like(labels)
    for i, n in enumerate(cut_frontier_nodes):
        for x in n.get_leaves():
            pred_labels[x.uid] = i
    
    # compute metrics
    fits = np.sum([
        np.sum(sparse_agglom_rep(sp.vstack((points[a], points[b]))))
            for _, a, b in constraints
    ])
    hg_tree = hg.Tree(struct_node_list)
    dp = hg.dendrogram_purity(hg_tree, labels)
    adj_rand_idx = adj_rand(pred_labels, labels)
    adj_mut_info = adj_mi(pred_labels, labels)

    metrics = {
        '# constraints': len(constraints),
        'fits' : fits,
        'pred_k' : len(cut_frontier_nodes),
        'dp' : round(dp, 4),
        '# maximally_pure_mergers' : len(maximally_pure_mergers),
        'adj_rand_idx' : round(adj_rand_idx, 4),
        'adj_mut_info' : round(adj_mut_info, 4),
        'cut_obj_score' : round(cut_obj_score, 4)
    }

    return pred_canon_ents, pred_labels, pred_tree_nodes, metrics


def gen_constraint(opt,
                   mention_labels,
                   pred_labels,
                   constraints,
                   num_to_generate=1):

    gold_pw_cc = ((mention_labels[:, None] == mention_labels[None, :])
                    ^ np.eye(mention_labels.size).astype(bool))
    pred_pw_cc = ((pred_labels[:, None] == pred_labels[None, :])
                    ^ np.eye(pred_labels.size).astype(bool))
    assert gold_pw_cc.shape == pred_pw_cc.shape
    candidate_constraints = np.where(np.triu(gold_pw_cc ^ pred_pw_cc, k=1))
    candidate_constraints = list(zip(*candidate_constraints))
    random.shuffle(candidate_constraints)
    
    for a, b in candidate_constraints[:num_to_generate]:
        if a == b:
            embed()
            exit()
        if gold_pw_cc[a, b]:
            constraints.append((np.inf, a, b))
        else:
            constraints.append((-np.inf, a, b))

    return constraints


def compute_idf(points):
    num_points = points.shape[0]
    doc_freq = np.sum(points.astype(bool).astype(int), axis=0)
    return np.log((num_points + 1) / (doc_freq + 1)) + 1


def run_mock_ml_sl(opt,
                   gold_entities,
                   mentions,
                   mention_labels,
                   sim_func):
    num_points = mentions.shape[0]
    constraints = []

    idf = compute_idf(mentions)
    mentions_normd = get_tfidf_normd(mentions, idf)

    # construct tree node objects for leaves
    leaves = [TreeNode(i, m_rep, label=lbl)
        for i, (m_rep, lbl) in enumerate(zip(mentions_normd, mention_labels))]

	## TESTING: generate some fake constraints
    #pos_idxs = np.where((mention_labels[:, None] == mention_labels[None, :]) ^ np.eye(mention_labels.size).astype(bool))
    #pos_edges = list(zip(*pos_idxs))
    #random.shuffle(pos_edges)

    #neg_idxs = np.where((mention_labels[:, None] != mention_labels[None, :]))
    #neg_edges = list(zip(*neg_idxs))
    #random.shuffle(neg_edges)

    #constraints.extend([(np.inf, a, b) for a, b in pos_edges[:5]])
    #constraints.extend([(-np.inf, a, b) for a, b in neg_edges[:5]])
    #random.shuffle(constraints)

    for r in range(opt.max_rounds):
        logger.debug('*** START - Clustering Points ***')
        # cluster the points
        out = cluster_points(
            opt,
            leaves,
            mention_labels,
            sim_func,
            constraints,
            opt.cost_per_cluster
        )
        pred_canon_ents, pred_labels, pred_tree_nodes, metrics = out
        logger.debug('*** END - Clustering Points ***')

        logger.info("round: {} - metrics: {}".format(r, metrics))
        if metrics['adj_rand_idx'] == 1.0:
            logger.info("perfect clustering reached in {} rounds".format(r))
            break

        # generate constraints every `iters` round
        iters = 1
        if r % iters == 0: 
            logger.debug('*** START - Generating Constraints ***')
            # generate constraints and viable places given predictions
            constraints = gen_constraint(
                opt,
                mention_labels,
                pred_labels,
                constraints,
                num_to_generate=opt.num_constraints_per_round
            )
            logger.debug('*** END - Generating Constraints ***')

            ## NOTE: JUST FOR TESTING
            #constraints = [(2*ent - 1) for ent in gold_entities]



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

    # real dataset
    parser.add_argument("--data_file", default=None, type=str,
                        help="preprocessed data pickle file in data_dir")

    # opts for building synthetic data
    parser.add_argument('--num_entities', type=int, default=2,
                        help="number of entities to generate when generating"\
                             "synthetic data")
    parser.add_argument('--num_mentions', type=int, default=10,
                        help="number of mentions to generate when generating"\
                             "synthetic data")
    parser.add_argument('--data_dim', type=int, default=16,
                        help="number of possible features (i.e. dimension of"\
                             "vector representation of points")
    parser.add_argument('--entity_noise_prob', type=float, default=0.2,
                        help="probability of noise features added to entity")
    parser.add_argument('--mention_sample_prob', type=float, default=0.7,
                        help="proportion of entity features added to mention")

    parser.add_argument('--cost_per_cluster', type=float, default=0.5,
                        help="lambda hyperparameter from paper")

    parser.add_argument('--sim_func', type=str,
                        choices=['cosine', 'jaccard'], default='cosine',
                        help="similarity function for clustering")
    parser.add_argument('--cluster_obj_reps', type=str,
                        choices=['raw', 'transformed'], default='raw',
                        help="which reps to use in tree cutting")
    parser.add_argument('--constraint_strength', type=int, default=1,
                        help="1/2 the max number features in gen constraint")
    parser.add_argument('--super_compat_score', action='store_true',
                        help="Enables super compatibility score when perfect")

    parser.add_argument('--max_rounds', type=int, default=100,
                        help="number of rounds to generate feedback for")
    parser.add_argument('--num_constraints_per_round', type=int, default=1,
                        help="number of constraints to generate per round")

    opt = parser.parse_args()

    # check to make sure there are no issues with the specified opts
    check_opt(opt)

    return opt


def check_opt(opt):
    # TODO: do a bunch of checks on the options
    pass


def set_sim_func(opt):
    sim_func = None
    if opt.sim_func == 'cosine':
        sim_func = cos_sim
    elif opt.sim_func == 'jaccard':
        sim_func = jaccard_sim
    assert sim_func is not None
    return sim_func


def drop_empty_columns(csr_mx):
    csc_mx = csr_mx.tocsc()
    csc_mx = csc_mx[:, np.diff(csc_mx.indptr) != 0]
    return csc_mx.tocsr()


def main():
    # get command line options
    opt = get_opt()

    # initialize the experiment
    initialize_exp(opt)

    if opt.data_file is not None:
        # get real data
        with open('{}/{}'.format(opt.data_dir, opt.data_file), 'rb') as f:
            gold_entities, mentions, mention_labels = pickle.load(f)
    else:
        # get or create the synthetic data
        data_fname = '{}/synth_data-{}_{}_{}_{}_{}-{}.pkl'.format(
            opt.data_dir,
            opt.num_entities,
            opt.num_mentions,
            opt.data_dim,
            opt.entity_noise_prob,
            opt.mention_sample_prob,
            opt.seed
        )
        if not os.path.exists(data_fname):
            with open(data_fname, 'wb') as f:
                gold_entities, mentions, mention_labels = gen_data(opt)
                pickle.dump((gold_entities, mentions, mention_labels), f)
        else:
            with open(data_fname, 'rb') as f:
                gold_entities, mentions, mention_labels = pickle.load(f)

    # drop empty columns from mentions and entities
    gold_entities = drop_empty_columns(gold_entities)
    mentions = drop_empty_columns(mentions)

    # declare similarity and compatibility functions with function pointers
    assert opt.sim_func == 'cosine' # TODO: support more sim funcs
    sim_func = set_sim_func(opt)

    # run the core function
    run_mock_ml_sl(
        opt, gold_entities, mentions, mention_labels, sim_func
    )


if __name__ == '__main__':
    main()

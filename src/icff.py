import os
import copy
import time
import pickle
import random
import logging
import argparse
from collections import deque, defaultdict
from heapq import heappop, heappush, heapify
from functools import reduce
from itertools import product
from tqdm import tqdm, trange

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix
from scipy.special import log_softmax

from sklearn.metrics import adjusted_rand_score as adj_rand
from sklearn.metrics import adjusted_mutual_info_score as adj_mi

import higra as hg
from sparse_dot_mkl import dot_product_mkl

from assign import greedy_assign
from compat_func import raw_overlap, transformed_overlap
from data import gen_data
from sim_func import dot_prod, jaccard_sim, cos_sim
from tree_ops import constraint_compatible_nodes
from tree_node import TreeNode
from utils import (MIN_FLOAT,
                   MAX_FLOAT,
                   InvalidAgglomError,
                   initialize_exp,
                   get_tfidf_normd,
                   sparse_agglom_rep,
                   get_nil_rep,
                   get_constraint_incompat)

from IPython import embed


logger = logging.getLogger(__name__)


def greedy_level_set_assign(viable_placements, incompat_mx):
    # Setup:
    # - queue for every constraint (based on sorted list of viable placements)
    # - heap of proposed assignments for unassigned constraints
    # - list(?) of chosen constraints and assignments
    num_constraints = len(viable_placements)

    running_vp = copy.deepcopy(viable_placements)

    # note this is a min-heap so we negate the score
    to_pick_heap = [(i, *d.popleft()) for i, d in enumerate(running_vp)]
    to_pick_heap = [(-s, (c, t)) for c, s, t in to_pick_heap]
    heapify(to_pick_heap)

    picked_cidxs = np.array([], dtype=int)
    picked_nidxs = np.array([], dtype=int)
    picked_scores = np.array([], dtype=int)

    valid_soln = True
    while len(picked_cidxs) < num_constraints:
        s, (c, n) = heappop(to_pick_heap)

        if len(picked_cidxs) == 0:
            picked_cidxs = np.append(picked_cidxs, int(c))
            picked_nidxs = np.append(picked_nidxs, int(n))
            picked_scores = np.append(picked_scores, -int(s))
            continue

        c_incompat_mask = incompat_mx[c, picked_cidxs]
        if np.any(picked_nidxs[c_incompat_mask] == n) :
            # if incompatible do the following
            try:
                _s, _n = running_vp[c].popleft()
            except:
                valid_soln = False
                break
            c_next_best = (-_s, (c, _n))
            heappush(to_pick_heap, c_next_best)
        else:
            picked_cidxs = np.append(picked_cidxs, int(c))
            picked_nidxs = np.append(picked_nidxs, int(n))
            picked_scores = np.append(picked_scores, -int(s))

    return np.sum(picked_scores), valid_soln


def sparse_update(full, mask, new):
    return sp.vstack((full[mask], new))

def dense_update(full, mask, new):
    return np.concatenate((full[mask], np.array([new])))


def custom_hac(opt,
               raw_points,
               transformed_points,
               idf,
               constraints,
               incompat_mx,
               compat_func):

    # constants
    num_points = raw_points.shape[0]
    num_constraints = len(constraints)

    # initialize level sets
    raw_level_set = raw_points.astype(float)
    transformed_level_set = transformed_points.astype(float)

    # bookkepping
    Xi = sp.vstack(constraints) if len(constraints) > 0 else None
    uids = np.arange(num_points)
    num_leaves = np.ones_like(uids)
    Z = np.empty((num_points-1, 4))

    # for computing best cut
    best_cut_score = MIN_FLOAT
    best_cut = copy.deepcopy(uids)
    cluster_ids = np.arange(num_points)
    intra_cluster_energies = np.ones_like(cluster_ids) * (1 / num_points)

    # pre-compute constraint_scores
    raw_points_normd = get_tfidf_normd(raw_points, idf)
    if num_constraints > 0:
        constraint_scores = compat_func(
            raw_points_normd,
            Xi,
            num_points if opt.super_compat_score else 1
        )
        prev_solns = {}

    # build initial similarity matrix
    transformed_level_set_normd = get_tfidf_normd(
        transformed_level_set.multiply(transformed_level_set > 0), idf
    )
    sim_mx = dot_product_mkl(
        transformed_level_set_normd, transformed_level_set_normd.T, dense=True
    )

    # whether or not we can cut anymore
    invalid_cut = False

    for r in trange(num_points-1):
        # ignore diag
        sim_mx[tuple([np.arange(transformed_level_set.shape[0])]*2)] = -np.inf

        # get next agglomeration
        while True:
            agglom_coord = np.where(sim_mx == sim_mx.max())
            agglom_coord = tuple(map(lambda x : x[0:1], agglom_coord))
            agglom_ind = np.array(list(map(lambda x : x[0], agglom_coord)))

            agglom_mask = np.zeros_like(uids, dtype=bool)
            agglom_mask[agglom_ind] = True

            if sim_mx[agglom_coord].item() > MIN_FLOAT:
                try:
                    transformed_agglom_rep = sparse_agglom_rep(
                        transformed_level_set[agglom_mask]
                    )
                    raw_agglom_rep = sparse_agglom_rep(
                        raw_level_set[agglom_mask]
                    )
                except InvalidAgglomError:
                    sim_mx[agglom_coord] = MIN_FLOAT
                    continue
            else:
                transformed_agglom_rep = get_nil_rep(
                    rep_dim=transformed_level_set.shape[1]
                )
                raw_agglom_rep = get_nil_rep(rep_dim=raw_level_set.shape[1])
            break
            
        # sanity check
        assert np.sum(agglom_mask) == 2

        # update data structures
        linkage_score = sim_mx[agglom_coord]
        not_agglom_mask = ~agglom_mask
        agglom_num_leaves = sum([num_leaves[x] for x in agglom_ind])

        # update linkage matrix
        Z[r] = np.array(
            [float(uids[agglom_ind[0]]),
             float(uids[agglom_ind[1]]),
             float(linkage_score),
             float(agglom_num_leaves)]
        )

        # get tfidf vectors for agglom reps
        transformed_agglom_rep_normd = get_tfidf_normd(
            transformed_agglom_rep.multiply(transformed_agglom_rep > 0), idf
        )
        raw_agglom_rep_normd = get_tfidf_normd(raw_agglom_rep, idf)

        # update level sets
        transformed_level_set = sparse_update(
            transformed_level_set, not_agglom_mask, transformed_agglom_rep
        )
        raw_level_set = sparse_update(
            raw_level_set, not_agglom_mask, raw_agglom_rep
        )
        transformed_level_set_normd = sparse_update(
            transformed_level_set_normd,
            not_agglom_mask,
            transformed_agglom_rep_normd
        )

        # update sim_mx
        num_untouched = np.sum(not_agglom_mask)
        sim_mx = sim_mx[not_agglom_mask[:,None] & not_agglom_mask[None,:]]
        sim_mx = sim_mx.reshape(num_untouched, num_untouched)
        sim_mx = np.concatenate(
            (sim_mx, np.ones((1, num_untouched)) * -np.inf), axis=0
        )
        new_sims = dot_product_mkl(
            transformed_level_set_normd,
            transformed_agglom_rep_normd.T,
            dense=True
        )
        sim_mx = np.concatenate((sim_mx, new_sims), axis=1)

        # update cluster_ids
        next_uid = np.max(uids) + 1
        new_cluster_mask = np.isin(cluster_ids, uids[agglom_mask])
        cluster_ids[new_cluster_mask] = next_uid

        # update uids list
        uids = dense_update(uids, not_agglom_mask, next_uid)

        # update num_leaves list
        num_leaves = dense_update(num_leaves, not_agglom_mask, agglom_num_leaves)

        # don't need to evaluate cut because constraints cannot be satisfied
        invalid_cut = invalid_cut or (linkage_score <= MIN_FLOAT)
        if invalid_cut:
            continue

        # update intra cluster energies
        agglom_energy = np.sum(
            dot_product_mkl(
                raw_points_normd[new_cluster_mask],
                raw_agglom_rep_normd.T,
                dense=True
            )
        )
        agglom_energy /= num_points
        intra_cluster_energies = dense_update(
            intra_cluster_energies, not_agglom_mask, agglom_energy
        )

        # compute best assignment of constraints to level_set
        assign_score = 0
        if num_constraints > 0:
            new_constraint_scores = compat_func(
                raw_agglom_rep_normd,
                Xi,
                num_points if opt.super_compat_score else 1
            )
            constraint_scores = np.concatenate(
                (constraint_scores[not_agglom_mask], new_constraint_scores),
                axis=0
            )

            node_idxs, constraint_idxs = np.where(constraint_scores > 0)
            scores = constraint_scores[(node_idxs, constraint_idxs)]

            viable_placements = [
                sorted([(scores[i], node_idxs[i]) 
                            for i in np.where(constraint_idxs == cuid)[0]], 
                        key=lambda x : x[0], reverse=True)
                    for cuid in range(num_constraints)
            ]
            viable_placements = [deque(l) for l in viable_placements
                                    if len(l) > 0]

            if len(viable_placements) < num_constraints:
                invalid_cut = True
                continue

            assign_score, valid_soln = greedy_level_set_assign(
                viable_placements, incompat_mx
            )

            if not valid_soln:
                invalid_cut = True
                continue

            if opt.compat_agg == 'avg':
                assign_score /= num_constraints

        cut_score = np.sum(intra_cluster_energies)\
                  - (opt.cost_per_cluster * intra_cluster_energies.size)\
                  + assign_score

        if raw_level_set.shape[0] == 1:
            continue

        if cut_score >= best_cut_score:
            #logger.debug((np.sum(intra_cluster_energies), 
            #              opt.cost_per_cluster * intra_cluster_energies.size,
            #              assign_score))
            best_cut_score = cut_score
            best_cut = copy.deepcopy(uids)

    # sanity check
    assert raw_level_set.shape[0] == 1 and transformed_level_set.shape[0] == 1

    return Z, best_cut, best_cut_score


def cluster_points(opt,
                   raw_points,
                   transformed_points,
                   idf,
                   leaf_nodes,
                   labels,
                   constraints,
                   compat_func):

    # pull out all of the points
    num_points = raw_points.shape[0]
    num_constraints = len(constraints)

    # compute constraint incompatibility matrix
    incompat_mx = get_constraint_incompat(constraints)

    # run clustering and produce the linkage matrix
    Z, best_cut, cut_obj_score = custom_hac(
        opt,
        raw_points,
        transformed_points,
        idf,
        constraints,
        incompat_mx,
        compat_func
    )

    # build the tree
    logger.debug('Constructing the tree')
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
    fits = int(np.sum([xi.size for xi in constraints]))
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


def gen_constraint_cheat(opt,
                         gold_entities,
                         pred_canon_ents,
                         pred_tree_nodes,
                         feat_freq,
                         constraints,
                         sim_func,
                         num_to_generate=1):

    assert num_to_generate == 1

    # maximally pure mergers
    maximally_pure_mergers = [n for n in pred_tree_nodes
            if n.label is not None and n.parent.label is None]
    #random.shuffle(maximally_pure_mergers)

    # TESTING: fixed target pure merger (the pure merger with mention 0 in it
    maximally_pure_mergers = [pm for pm in maximally_pure_mergers 
            if len([x for x in pm.get_leaves() if x.uid == 0]) > 0]

    logger.debug('tgt pm {} - num leaves: {}'.format(
            maximally_pure_mergers[0].uid,
            len(maximally_pure_mergers[0].get_leaves())
        )
    )

    pure_merger_iter = iter(maximally_pure_mergers)

    
    # create constraints using maximally pure mergers
    num_gen_constraints = 0
    while num_gen_constraints < num_to_generate:
        pure_merger = next(pure_merger_iter)
        gold_ent_rep = gold_entities[pure_merger.label]

        pm_transformed_rep = (pure_merger.transformed_rep > 0).astype(float)
        par_transformed_rep = (pure_merger.parent.transformed_rep > 0).astype(float)

        neg_feats = ((par_transformed_rep - gold_ent_rep) > 0).astype(float)
        pos_feats = ((gold_ent_rep - pm_transformed_rep) > 0).astype(float)

        if pos_feats.tocoo().col.size == 0:
            continue

        logger.debug('Generating constraint for node: {}'.format(pure_merger.uid))

        in_idxs = pm_transformed_rep.tocoo().col
        pos_idxs = np.random.choice(pos_feats.tocoo().col, size=(1,))
        #pos_idxs = pos_feats.tocoo().col
        #neg_idxs = np.random.choice(neg_feats.tocoo().col, size=(1,))
        neg_idxs = neg_feats.tocoo().col

        constraint_cols = np.concatenate((pos_idxs, in_idxs, neg_idxs), axis=0)
        constraint_data = [1] * (pos_idxs.size + in_idxs.size)\
                          + [-1] * neg_idxs.size
        constraint_rows = np.zeros_like(constraint_cols)

        new_constraint = csr_matrix(
            (constraint_data, (constraint_rows, constraint_cols)),
            shape=gold_ent_rep.shape,
            dtype=float
        )

        constraints.append(new_constraint)

        num_gen_constraints += 1

    return constraints


def gen_constraint(opt,
                   gold_entities,
                   pred_canon_ents,
                   pred_tree_nodes,
                   feat_freq,
                   constraints,
                   sim_func,
                   num_to_generate=1):

    totals = np.sum(feat_freq, axis=0)
    feat_freq_normd = feat_freq / totals

    for _ in trange(num_to_generate):
        # oracle feedback generation in the form of there-exists constraints
        pred_ent_idx = random.randint(0, pred_canon_ents.shape[0]-1)
        ff_pred_ent = pred_canon_ents[pred_ent_idx]
        feat_intersect = dot_product_mkl(
            ff_pred_ent.multiply(gold_entities), ff_pred_ent.T, dense=True
        ).reshape(-1,)
        is_ent_subsets = feat_intersect == np.sum(ff_pred_ent)
        num_ent_subsets = np.sum(is_ent_subsets)
        assert num_ent_subsets < 2

        # get the tgt entity for this constraint
        tgt_ent_idx = np.argmax(feat_intersect)
        tgt_gold_ent = gold_entities[tgt_ent_idx]

        while True:

            # indices of features in predicted ent domain
            ff_pred_ent_domain = ff_pred_ent.tocoo().col

            # sample "pos"-features
            pos_pred_domain = np.array(
                list(
                    set(tgt_gold_ent.tocoo().col).difference(
                        set(ff_pred_ent_domain)
                    )
                )
            )
            pos_idxs = np.array([])
            if pos_pred_domain.size > 0:
                pos_feat_dist = np.exp(
                    log_softmax(
                        feat_freq_normd[tgt_ent_idx, pos_pred_domain]
                    )
                )
                pos_idxs = np.random.choice(
                    pos_pred_domain,
                    #size=min(opt.constraint_strength, pos_pred_domain.size),
                    size=1,
                    replace=False,
                    p=pos_feat_dist
                )

            # sample "in"-features
            in_pred_domain = ff_pred_ent.multiply(tgt_gold_ent).tocoo().col
            in_feat_dist = np.exp(
                log_softmax(
                    feat_freq_normd[tgt_ent_idx][in_pred_domain]
                )
            )
            in_idxs = np.random.choice(
                in_pred_domain,
                size=min(2*opt.constraint_strength - pos_idxs.size,
                         in_pred_domain.size),
                replace=False,
                p=in_feat_dist
            )

            # sample "neg"-features
            if num_ent_subsets == 0:
                neg_pred_domain = np.array(
                    list(
                        set(ff_pred_ent_domain).difference(
                            set(in_pred_domain)
                        )
                    )
                )
                neg_feat_dist = np.exp(
                    log_softmax(
                        np.sum(feat_freq[:, neg_pred_domain], axis=0)
                    )
                )
            else:
                not_gold_mask = ~tgt_gold_ent.toarray().reshape(-1,).astype(bool)
                neg_pred_domain = np.where(not_gold_mask)[0]
                corr_weights = np.sum(
                    feat_freq_normd[:, in_pred_domain], axis=1
                )
                neg_feat_dist = np.exp(
                    log_softmax(
                        np.sum(
                            feat_freq_normd[:, not_gold_mask]\
                                    * corr_weights.reshape(-1, 1), axis=0
                        )
                    )
                )
            neg_idxs = np.random.choice(
                neg_pred_domain,
                #size=min([opt.constraint_strength,
                #          neg_pred_domain.size,
                #          np.sum(neg_feat_dist > 0)]),
                size=1,
                replace=False,
                p=neg_feat_dist
            )

            # create constraint
            ff_constraint_cols = np.concatenate((pos_idxs, in_idxs, neg_idxs), axis=0)
            ff_constraint_data = [1] * (pos_idxs.size + in_idxs.size)\
                                 + [-1] * neg_idxs.size
            ff_constraint_rows = np.zeros_like(ff_constraint_cols)
            ff_constraint = csr_matrix(
               (ff_constraint_data, (ff_constraint_rows, ff_constraint_cols)),
               shape=tgt_gold_ent.shape,
               dtype=float
            )

            # check to make sure this constraint doesn't exist yet
            already_exists = np.any([
                (ff_constraint != xi).nnz == 0 for xi in constraints
            ])

            if not already_exists:
                break

        # add constraint to running list
        constraints.append(ff_constraint)

    return constraints


def get_feat_freq(mentions, mention_labels):
    counts_rows = []
    for lbl_id in np.unique(mention_labels):
        counts_rows.append(np.sum(mentions[mention_labels == lbl_id], axis=0))
    counts = np.asarray(np.concatenate(counts_rows))
    return counts


def compute_idf(points):
    num_points = points.shape[0]
    doc_freq = np.sum(points.astype(bool).astype(int), axis=0)
    return np.log((num_points + 1) / (doc_freq + 1)) + 1


def get_assign_metrics(leaf2constraints, labels, gold_entities, constraints):
    constraint2num_leaves = defaultdict(int)
    num_wrong_assignments = 0
    num_assignments = 0
    for muids, cuids in leaf2constraints.items():
        ent_id = labels[muids]
        for cuid in cuids:
            constraint2num_leaves[cuid] += 1
            num_assignments += 1
            if np.sum(np.abs(constraints[cuid]-gold_entities[ent_id]) == 2) > 0:
                num_wrong_assignments += 1

    num_lvs = list(constraint2num_leaves.values())

    ret_str = "# lvs / constraint --  min: {} ; max: {} ; mean: {}\n"\
              "frac wrong node assign : {} / {}".format(
        np.min(num_lvs),
        np.max(num_lvs),
        np.mean(num_lvs),
        num_wrong_assignments,
        num_assignments
    )

    return ret_str


def run_mock_icff(opt,
                  gold_entities,
                  mentions,
                  labels,
                  sim_func,
                  compat_func):

    num_points = mentions.shape[0]
    constraints = []

    # TODO: come back to maybe deleting this?
    feat_freq = get_feat_freq(mentions, labels)

    # construct tree node objects for leaves
    leaf_nodes = [TreeNode(i, m_rep, label=lbl)
                    for i, (m_rep, lbl) in enumerate(zip(mentions, labels))]

    ## NOTE: JUST FOR TESTING
    #constraints = [csr_matrix(2*ent - 1, dtype=float)
    #                    for ent in gold_entities.toarray()]
    #for i, xi in enumerate(constraints):
    #    first_compat_idx = np.where(labels == i)[0][0]
    #    first_compat_mention = mentions[first_compat_idx]
    #    transformed_rep = sparse_agglom_rep(
    #        sp.vstack((first_compat_mention, xi))
    #    )
    #    leaf_nodes[i].transformed_rep = transformed_rep

    raw_points = sp.vstack([x.raw_rep for x in leaf_nodes])
    idf = compute_idf(raw_points)

    for r in range(opt.max_rounds+1):
        # points are sparse count vectors
        transformed_points = sp.vstack([x.transformed_rep for x in leaf_nodes])

        logger.debug('*** START - Clustering Points ***')
        # cluster the points
        if r == 0:
            with open('cluster_out_r0.pkl', 'rb') as f:
                out = pickle.load(f)
        else:
            out = cluster_points(opt,
                                 raw_points,
                                 transformed_points,
                                 idf,
                                 leaf_nodes,
                                 labels,
                                 constraints,
                                 compat_func)
        pred_canon_ents, pred_labels, pred_tree_nodes, metrics = out
        logger.debug('*** END - Clustering Points ***')

        logger.info("round: {} - metrics: {}".format(r, metrics))

        # keeping track of the tgt max pm
        maximally_pure_mergers = [n for n in pred_tree_nodes
                if n.label is not None and n.parent.label is None]
        maximally_pure_mergers = [pm for pm in maximally_pure_mergers 
                if len([x for x in pm.get_leaves() if x.uid == 0]) > 0]

        logger.debug('tgt pm {} - num leaves: {}'.format(
                maximally_pure_mergers[0].uid,
                len(maximally_pure_mergers[0].get_leaves())
            )
        )

        # for debugging
        if r == 1:
            #supset_placement_leaves = [n.uid for n in pred_tree_nodes
            #        if set(placement_leaves).issubset(set([l.uid for l in n.get_leaves()]))]

            embed()
            exit()

        if metrics['adj_rand_idx'] == 1.0:
            logger.info("perfect clustering reached in {} rounds".format(r))
            break

        # generate constraints every `iters` round
        iters = 1
        if r % iters == 0: 
            logger.debug('*** START - Generating Constraints ***')
            # generate constraints and viable places given predictions
            #constraints = gen_constraint(
            #    opt,
            #    gold_entities,
            #    pred_canon_ents,
            #    pred_tree_nodes,
            #    feat_freq,
            #    constraints,
            #    sim_func,
            #    num_to_generate=opt.num_constraints_per_round
            #)
            constraints = gen_constraint_cheat(
                opt,
                gold_entities,
                pred_canon_ents,
                pred_tree_nodes,
                feat_freq,
                constraints,
                sim_func,
                num_to_generate=opt.num_constraints_per_round
            ) 
            logger.debug('*** END - Generating Constraints ***')

        ## NOTE: JUST FOR TESTING
        #constraints = [csr_matrix(2*ent - 1, dtype=float)
        #                    for ent in gold_entities.toarray()]

        logger.debug('*** START - Computing Viable Placements ***')
        viable_placements = constraint_compatible_nodes(
            opt, pred_tree_nodes, idf, constraints, compat_func, num_points
        )
        logger.debug('*** END - Computing Viable Placements ***')

        logger.debug('*** START - Assigning Constraints ***')
        # solve structured prediction problem of jointly placing the constraints
        placements_out = greedy_assign(
            pred_tree_nodes,
            constraints,
            viable_placements,
        )
        logger.debug('*** END - Assigning Constraints ***')

        #### TEST: for debugging
        #compressed_assign = [(s, n.uid) for s, n in viable_placements[0]]
        #placement_leaves = [n.uid for n in pred_tree_nodes[3785].get_leaves()]

        #with open('debug_cluster_lvs.pkl', 'rb') as f:
        #    next_round_sib_lvs = pickle.load(f)

        #supset_placement_leaves = [n.uid for n in pred_tree_nodes
        #        if set(next_round_sib_lvs).issubset(set([l.uid for l in n.get_leaves()]))]

        ####

        logger.debug('Constraint assignments: {}'.format(placements_out))

        logger.debug('*** START - Projecting Assigned Constraints ***')
        # reset all leaf transformed_rep's
        logger.debug('Reseting leaves')
        for node in leaf_nodes:
            node.transformed_rep = copy.deepcopy(node.raw_rep)

        # transform the placements out to leaf2constraints
        logger.debug('Expanding placements to leaves')
        nuid2luids = {n.uid : [x.uid for x in n.get_leaves()]
                         for n in pred_tree_nodes}
        leaf2constraints = defaultdict(set)
        for nuid, cuids in placements_out.items():
            for luid in nuid2luids[nuid]:
                leaf2constraints[luid].update(cuids)

        # get and display assignment metrics
        assignment_metrics = get_assign_metrics(
            leaf2constraints, labels, gold_entities, constraints
        )
        logger.info('assign metrics : {}'.format(assignment_metrics))

        # project resolved constraint placements to leaves
        logger.debug('Projecting constraints to expanded placements')
        for nuid, cuids in leaf2constraints.items():
            reps = [pred_tree_nodes[nuid].transformed_rep]\
                    + [constraints[cuid] for cuid in cuids]
            pred_tree_nodes[nuid].transformed_rep = sparse_agglom_rep(
                sp.vstack(reps)
            )

        logger.debug('*** END - Projecting Assigned Constraints ***')

    embed()
    exit()



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
                        help="proportion of entity features added to mention")


    parser.add_argument('--cluster_obj_reps', type=str,
                        choices=['raw', 'transformed'], default='raw',
                        help="which reps to use in tree cutting")
    parser.add_argument('--compat_agg', type=str,
                        choices=['avg', 'sum'], default='avg',
                        help="how to aggregate constraint compatibility in obj")
    parser.add_argument('--sim_func', type=str,
                        choices=['cosine', 'jaccard'], default='cosine',
                        help="similarity function for clustering")
    parser.add_argument('--compat_func', type=str,
                        choices=['raw', 'transformed'], default='raw',
                        help="compatibility function constraint satisfaction")
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


def set_compat_func(opt):
    compat_func = None
    if opt.compat_func == 'raw':
        compat_func = raw_overlap
    elif opt.compat_func == 'transformed':
        compat_func = transformed_overlap
    assert compat_func is not None
    return compat_func


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
                gold_entities = csr_matrix(gold_entities)
                mentions = csr_matrix(mentions)
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
    compat_func = set_compat_func(opt)

    # run the core function
    run_mock_icff(
        opt, gold_entities, mentions, mention_labels, sim_func, compat_func
    )


if __name__ == '__main__':
    main()

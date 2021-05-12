import os
import copy
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
from match import match_constraints, lca_check, viable_match_check
from sim_func import dot_prod, jaccard_sim, cos_sim
from tree_ops import constraint_compatible_nodes
from tree_node import TreeNode
from utils import (MIN_FLOAT,
                   MAX_FLOAT,
                   InvalidAgglomError,
                   initialize_exp,
                   sparse_agglom_rep,
                   get_nil_rep)

from IPython import embed


logger = logging.getLogger(__name__)


def custom_hac(points, sim_func):

    # FIXME: sim_func is unused

    level_set = points.astype(float)
    uids = np.arange(level_set.shape[0])
    num_leaves = np.ones_like(uids)
    Z = []

    level_set_normd = normalize(level_set, norm='l2', axis=1)
    sim_mx = dot_product_mkl(level_set_normd, level_set_normd.T, dense=True)

    for _ in trange(level_set.shape[0]-1):
        sim_mx[tuple([np.arange(level_set.shape[0])]*2)] = -np.inf # ignore diag

        # get next agglomeration
        while True:
            agglom_coord = np.where(sim_mx == sim_mx.max())
            agglom_coord = tuple(map(lambda x : x[0:1], agglom_coord))
            agglom_ind = np.array(list(map(lambda x : x[0], agglom_coord)))

            agglom_mask = np.zeros_like(uids, dtype=bool)
            agglom_mask[agglom_ind] = True

            if sim_mx[agglom_coord].item() > MIN_FLOAT:
                try:
                    agglom_rep = sparse_agglom_rep(level_set[agglom_mask])
                except InvalidAgglomError:
                    sim_mx[agglom_coord] = MIN_FLOAT
                    continue
            else:
                agglom_rep = get_nil_rep(rep_dim=level_set.shape[1])
            break
            
        assert np.sum(agglom_mask) == 2

        # update data structures
        linkage_score = sim_mx[agglom_coord]
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
        agglom_rep_normd = normalize(agglom_rep, norm='l2', axis=1)
        level_set_normd = sp.vstack(
            (level_set_normd[not_agglom_mask], agglom_rep_normd)
        )
        new_sims = dot_product_mkl(
            level_set_normd, agglom_rep_normd.T, dense=True
        )
        sim_mx = np.concatenate((sim_mx, new_sims), axis=1)

        # update uids list
        next_uid = np.max(uids) + 1
        uids = np.concatenate(
            (uids[not_agglom_mask], np.array([next_uid]))
        )
        # update num_leaves list
        num_leaves = np.concatenate(
            (num_leaves[not_agglom_mask], np.array([agglom_num_leaves]))
        )

    # sanity check
    assert level_set.shape[0] == 1

    # return the linkage matrix
    Z = np.vstack(Z)

    return Z


def intra_subcluster_energy(opt, subcluster, sim_func, num_points):
    subcluster_leaves = subcluster.get_leaves()
    assert len(subcluster_leaves) > 0

    if opt.cluster_obj_reps == 'raw':
        reps = sp.vstack([n.raw_rep for n in subcluster_leaves])
    else:
        assert opt.cluster_obj_reps == 'transformed'
        reps = sp.vstack([n.transformed_rep for n in subcluster_leaves])

    canon_rep = sparse_agglom_rep(reps)
    canon_rep_normd = normalize(canon_rep, norm='l2', axis=1)
    reps_normd = normalize(reps, norm='l2', axis=1)
    rep_affinities = dot_product_mkl(reps_normd, canon_rep_normd.T, dense=True)
    return np.sum(rep_affinities) / num_points


def constraint_satisfaction(opt,
                            node,
                            compat_func,
                            constraints,
                            num_points,
                            num_constraints):

    constraints_satisfied = {}
    for i, xi in enumerate(constraints):
        compat_score = compat_func(
            node, xi, num_points if opt.super_compat_score else 1
        )
        if compat_score > 0:
            if opt.compat_agg == 'sum':
                constraints_satisfied[i] = compat_score
            else:
                assert opt.compat_agg == 'avg'
                constraints_satisfied[i] = compat_score / num_constraints
    return constraints_satisfied


def value_node(opt,
               node,
               sim_func,
               compat_func,
               constraints,
               incompat_mx,
               cost_per_cluster,
               num_points,
               num_constraints):

    # compute raw materials
    intra_energy = intra_subcluster_energy(opt, node, sim_func, num_points)
    intra_energy -= cost_per_cluster
    satisfy_energies = constraint_satisfaction(
        opt, node, compat_func, constraints, num_points, num_constraints
    )

    # fill the value map
    value_map = {tuple() : (intra_energy, [node], {})}
    if len(satisfy_energies) > 0:
        # resolve max subsets of compatible constraints
        cnstrt_idxs = list(satisfy_energies.keys())
        sub_incompat_mx = incompat_mx[np.ix_(cnstrt_idxs, cnstrt_idxs)]
        if np.any(sub_incompat_mx): # hard case: not every valid constraint is compatible
            # get masks of all valid maximal subsets from cnstrt_idxs
            max_ss_idxs = ~np.unique(
                sub_incompat_mx[np.any(sub_incompat_mx, axis=0)], axis=0
            )
            for ss in max_ss_idxs:
                ss_cnstrt_idxs = np.asarray(cnstrt_idxs)[np.where(ss)[0]]
                ss_cnstrt_energy = sum(
                    [satisfy_energies[k] for k in ss_cnstrt_idxs]
                )
                value_map[tuple(ss_cnstrt_idxs)] = (
                    intra_energy + ss_cnstrt_energy,
                    [node],
                    {k : satisfy_energies[k] for k in ss_cnstrt_idxs}
                )
        else: # easy case: every valid constraint is compatible
            agg_energy = sum(satisfy_energies.values())
            value_map[tuple(sorted(satisfy_energies.keys()))] = (
                intra_energy + agg_energy,
                [node],
                satisfy_energies
            )

    return value_map


def memoize_subcluster(opt,
                       node,
                       sim_func,
                       compat_func,
                       constraints,
                       incompat_mx,
                       cost_per_cluster,
                       num_points,
                       num_constraints):

    node_map = value_node(
        opt,
        node,
        sim_func,
        compat_func,
        constraints,
        incompat_mx,
        cost_per_cluster,
        num_points,
        num_constraints
    )

    if len(node.children) > 0:
        child_maps = [memoize_subcluster(opt,
                                         c, 
                                         sim_func,
                                         compat_func,
                                         constraints,
                                         incompat_mx,
                                         cost_per_cluster,
                                         num_points,
                                         num_constraints)
                        for c in node.children]
        assert len(child_maps) == 2 # restrict to binary trees for now

        # resolve child maps
        resolved_child_map = {}
        for l_key, r_key in product(*[list(m.keys()) for m in child_maps]):
            constraint_key = tuple(sorted(list(set(list(l_key)+list(r_key)))))
            if len(constraint_key) == len(l_key) + len(r_key):
                # non-overlap case
                max_val, _, _ = resolved_child_map.get(
                    constraint_key, (-np.inf, None, None)
                )
                l_val, l_cut, l_cnstrt = child_maps[0][l_key]
                r_val, r_cut, r_cnstrt = child_maps[1][r_key]
                this_val = l_val + r_val
                if this_val >= max_val:
                    resolved_child_map[constraint_key] = (
                        this_val,
                        l_cut + r_cut,
                        l_cnstrt | r_cnstrt
                    )
            else:
                # overlap case
                l_val, l_cut, l_cnstrt = child_maps[0][l_key]
                r_val, r_cut, r_cnstrt = child_maps[1][r_key]
                bare_val = (l_val - sum(l_cnstrt.values())
                            + r_val - sum(r_cnstrt.values()))
                # resolve overlap
                merged_cnstrt = {
                    k : max(i for i in (l_cnstrt.get(k), r_cnstrt.get(k)) if i)
                        for k in l_cnstrt.keys() | r_cnstrt.keys()
                }

                max_val, _, _ = resolved_child_map.get(
                    constraint_key, (-np.inf, None, None)
                )
                this_val = bare_val + sum(merged_cnstrt.values())
                if this_val >= max_val:
                    resolved_child_map[constraint_key] = (
                        this_val,
                        l_cut + r_cut,
                        merged_cnstrt
                    )

        # return map of max over merged keys of dicts
        for key, cut_rep in resolved_child_map.items():
            if key in node_map.keys():
                if cut_rep[0] > node_map[key][0]:
                    node_map[key] = resolved_child_map[key]
            else:
                node_map[key] = resolved_child_map[key]

    # check supersets
    node_map_keys = sorted(node_map.keys(), key=lambda k : len(k))
    for supset_key in node_map_keys:
        for subset_key in node_map_keys:
            if len(subset_key) >= len(supset_key):
                break
            if subset_key not in node_map.keys(): # we're deleting some of the keys
                continue

            # check to see whether or not we should delete subset_key
            supset_val, supset_cut, supset_cnstrt = node_map[supset_key]
            subset_val, subset_cut, subset_cnstrt = node_map[subset_key]
            supset_cmp_val = supset_val - sum(
                [v for k, v in supset_cnstrt.items() 
                    if k not in subset_cnstrt.keys()]
            )
            if supset_cmp_val < subset_val:
                continue
            else:
                # don't need subset key anymore
                del node_map[subset_key]

    return node_map
        

def get_opt_tree_cut(opt,
                     pred_tree_nodes,
                     sim_func,
                     compat_func,
                     constraints,
                     cost_per_cluster,
                     num_points,
                     num_constraints):

    incompat_mx = None
    if len(constraints) > 0:
        Xi = sp.vstack(constraints)
        extreme_constraints = copy.deepcopy(Xi)
        extreme_constraints.data *= np.inf
        incompat_mx = dot_product_mkl(
            extreme_constraints, extreme_constraints.T, dense=True
        )
        incompat_mx = (incompat_mx == -np.inf) | np.isnan(incompat_mx)

    # recursively compute value map of root node
    assert pred_tree_nodes[-1].parent is None
    root_value_map = memoize_subcluster(
        opt,
        pred_tree_nodes[-1],
        sim_func,
        compat_func,
        constraints,
        incompat_mx,
        cost_per_cluster,
        num_points,
        num_constraints
    )

    # pick max cut out of `root_value_map` (maximizing energy)
    max_cut_keys = [k for k in root_value_map.keys() 
                        if len(k) == len(constraints)]
    assert len(max_cut_keys) == 1
    max_cut_score, max_cut, _ = root_value_map[max_cut_keys[0]]

    return max_cut, max_cut_score


def cluster_points(opt,
                   leaf_nodes,
                   labels,
                   sim_func,
                   compat_func,
                   constraints,
                   cost_per_cluster):
    # pull out all of the points
    points = sp.vstack([x.transformed_rep for x in leaf_nodes])
    num_points = points.shape[0]
    num_constraints = len(constraints)

    # run clustering and produce the linkage matrix
    Z = custom_hac(points, sim_func)

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

    # find the best cut
    cut_frontier_nodes, cut_obj_score = get_opt_tree_cut(
        opt,
        pred_tree_nodes,
        sim_func,
        compat_func,
        constraints,
        cost_per_cluster,
        num_points,
        num_constraints
    )

    # the predicted entities canonicalization
    pred_canon_ents = sp.vstack(
        [n.raw_rep for n in cut_frontier_nodes]
    )

    # produce the predicted labels for leaves
    pred_labels = np.zeros_like(labels)
    for i, n in enumerate(cut_frontier_nodes):
        for x in n.get_leaves():
            pred_labels[x.uid] = i
    
    # compute metrics
    fits = np.sum([xi.size for xi in constraints])
    hg_tree = hg.Tree(struct_node_list)
    dp = hg.dendrogram_purity(hg_tree, labels)
    adj_rand_idx = adj_rand(pred_labels, labels)
    adj_mut_info = adj_mi(pred_labels, labels)

    metrics = {
        'pred_k' : len(cut_frontier_nodes),
        'fits' : fits,
        'dp' : round(dp, 4),
        'adj_rand_idx' : round(adj_rand_idx, 4),
        'adj_mut_info' : round(adj_mut_info, 4),
        'cut_obj_score' : round(cut_obj_score, 4)
    }

    return pred_canon_ents, pred_labels, pred_tree_nodes, metrics


def gen_constraint(opt,
                   gold_entities,
                   pred_canon_ents,
                   pred_tree_nodes,
                   constraints,
                   sim_func,
                   num_to_generate=1):

    for _ in range(num_to_generate):
        # oracle feedback generation in the form of there-exists constraints
        pred_ent_idx = random.randint(0, len(pred_canon_ents)-1)
        ff_pred_ent = pred_canon_ents[pred_ent_idx]
        feat_intersect = ff_pred_ent & gold_entities
        feat_subset = feat_intersect == ff_pred_ent
        is_ent_subsets = np.all(feat_subset, axis=1)
        num_ent_subsets = np.sum(is_ent_subsets)
        assert num_ent_subsets < 2
        if num_ent_subsets == 0:
            logger.debug('****** SPLIT CONSTRAINT ******')
            # split required
            tgt_ent_idx = np.argmax(np.sum(feat_intersect, axis=1))
        else:
            logger.debug('****** MERGE CONSTRAINT ******')
            assert num_ent_subsets == 1
            # merge required
            tgt_ent_idx = np.argmax(is_ent_subsets)

        tgt_gold_ent = gold_entities[tgt_ent_idx]

        while True:
            # init constraint
            ff_constraint = np.zeros_like(tgt_gold_ent)

            # sample "in"-features
            in_pred_domain = np.where(ff_pred_ent == 1)[0]
            np.random.shuffle(in_pred_domain)
            in_idxs = in_pred_domain[:opt.constraint_strength]

            # select 2nd closest gold entity
            arg_feat_overlap = np.argsort(np.sum(feat_intersect, axis=1))
            assert arg_feat_overlap[-1] == tgt_ent_idx
            neg_tgt_idx = arg_feat_overlap[-2]
            neg_tgt_ent = gold_entities[neg_tgt_idx]

            # sample "out"-features
            out_pos_feats = (tgt_gold_ent ^ ff_pred_ent) & (1-neg_tgt_ent)
            out_pred_domain = np.where(out_pos_feats)[0]
            np.random.shuffle(out_pred_domain)
            out_idxs = out_pred_domain[:opt.constraint_strength]

            # sample "neg"-features
            out_neg_feats = neg_tgt_ent & (1 - tgt_gold_ent)
            out_neg_domain = np.where(out_neg_feats)[0]
            np.random.shuffle(out_neg_domain)
            neg_idxs = out_neg_domain[:opt.constraint_strength]

            # fill in constraint
            ff_constraint[in_idxs] = 1
            ff_constraint[out_idxs] = 1
            ff_constraint[neg_idxs] = -1

            if not any([np.array_equal(ff_constraint, xi) for xi in constraints]):
                break

        constraints.append(ff_constraint)

    return constraints


def run_mock_icff(opt,
                  gold_entities,
                  mentions,
                  mention_labels,
                  sim_func,
                  compat_func):

    num_points = mentions.shape[0]
    constraints = []

    # construct tree node objects for leaves
    leaves = [TreeNode(i, m_rep) for i, m_rep in enumerate(mentions)]

    # NOTE: JUST FOR TESTING
    constraints = [csr_matrix(2*ent - 1, dtype=float)
                        for ent in gold_entities.toarray()]
    for i, xi in enumerate(constraints):
        first_compat_idx = np.where(mention_labels == i)[0][0]
        first_compat_mention = mentions[first_compat_idx]
        transformed_rep = sparse_agglom_rep(
            sp.vstack((first_compat_mention, xi))
        )
        leaves[i].transformed_rep = transformed_rep

    for r in range(opt.max_rounds):
        logger.debug('*** START - Clustering Points ***')
        # cluster the points
        out = cluster_points(
            opt,
            leaves,
            mention_labels,
            sim_func,
            compat_func,
            constraints,
            opt.cost_per_cluster
        )
        pred_canon_ents, pred_labels, pred_tree_nodes, metrics = out
        logger.debug('*** END - Clustering Points ***')

        logger.info("round: {} - metrics: {}".format(r, metrics))
        if metrics['adj_rand_idx'] == 1.0:
            logger.info("perfect clustering reached in {} rounds".format(r))
            break

        ## generate constraints every `iters` round
        #iters = 1
        #if r % iters == 0: 
        #    logger.debug('*** START - Generating Constraints ***')
        #    # generate constraints and viable places given predictions
        #    constraints = gen_constraint(
        #        opt,
        #        gold_entities,
        #        pred_canon_ents,
        #        pred_tree_nodes,
        #        constraints,
        #        sim_func,
        #        num_to_generate=opt.num_constraints_per_round
        #    )
        #    logger.debug('*** END - Generating Constraints ***')

        #    ## NOTE: JUST FOR TESTING
        #    #constraints = [(2*ent - 1) for ent in gold_entities]

        logger.debug('*** START - Computing Viable Placements ***')
        # update constraints and viable placements
        viable_placements = []
        for xi in constraints:
            compatible_nodes = constraint_compatible_nodes(
                opt, pred_tree_nodes, xi, compat_func, num_points
            )
            viable_placements.append(
                sorted(
                    compatible_nodes,
                    key=lambda x: (x[0], x[1].uid),
                    reverse=True
                )
            )
        logger.debug('*** END - Computing Viable Placements ***')

        logger.debug('*** START - Assigning Constraints ***')
        # solve structured prediction problem of jointly placing the constraints
        placements_out = match_constraints(
            constraints,
            viable_placements,
            lca_check,
            allow_no_match=False    # may change this arg later
        )
        assert placements_out is not None
        _, resolved_placements = placements_out
        logger.debug('*** END - Assigning Constraints ***')

        logger.debug('*** START - Projecting Assigned Constraints ***')
        # reset all leaf transformed_rep's
        for node in leaves:
            node.transformed_rep = copy.deepcopy(node.raw_rep)

        # project resolved constraint placements to leaves
        for xi, rp in zip(constraints, resolved_placements):
            for node in rp.get_leaves():
                node.transformed_rep |= xi
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

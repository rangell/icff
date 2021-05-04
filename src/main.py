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

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score as adj_rand
from sklearn.metrics import adjusted_mutual_info_score as adj_mi

from compat_func import raw_overlap, transformed_overlap
from data import gen_data
from match import match_constraints, lca_check, viable_match_check
from metrics import dendrogram_purity
from sim_func import dot_prod, jaccard_sim, cos_sim
from tree_ops import constraint_compatible_nodes
from tree_node import TreeNode
from utils import initialize_exp

from IPython import embed


logger = logging.getLogger(__name__)


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


def intra_subcluster_energy(subcluster, sim_func, num_points):
    subcluster_leaves = subcluster.get_leaves()
    assert len(subcluster_leaves) > 0
    reps = np.vstack([n.raw_rep for n in subcluster_leaves])
    canon_rep = reduce(lambda a, b : a | b, reps)[None, :]
    rep_affinities = sim_func(reps, canon_rep)
    return np.sum(rep_affinities) / num_points


def constraint_satisfaction(node, compat_func, constraints, num_constraints):
    constraints_satisfied = {}
    for i, xi in enumerate(constraints):
        compat_score = compat_func(node, xi)
        if compat_score > 0:
            constraints_satisfied[i] = compat_score / num_constraints
    return constraints_satisfied


def value_node(node,
               sim_func,
               compat_func,
               constraints,
               incompat_mx,
               cost_per_cluster,
               num_points,
               num_constraints):
    # compute raw materials
    intra_energy = intra_subcluster_energy(node, sim_func, num_points)
    intra_energy -= cost_per_cluster
    satisfy_energies = constraint_satisfaction(
        node, compat_func, constraints, num_constraints
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


def memoize_subcluster(node,
                       sim_func,
                       compat_func,
                       constraints,
                       incompat_mx,
                       cost_per_cluster,
                       num_points,
                       num_constraints):

    node_map = value_node(
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
        child_maps = [memoize_subcluster(c, 
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
        

def get_opt_tree_cut(pred_tree_nodes,
                     sim_func,
                     compat_func,
                     constraints,
                     cost_per_cluster,
                     num_points,
                     num_constraints):

    incompat_mx = None
    if len(constraints) > 0:
        Xi = np.vstack(constraints)
        incompat_mx = np.any((Xi[:, None, :] * Xi[None, :, :]) < 0, axis=-1)

    # recursively compute value map of root node
    assert pred_tree_nodes[-1].parent is None
    root_value_map = memoize_subcluster(
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


def cluster_points(leaf_nodes,
                   labels,
                   sim_func,
                   compat_func,
                   constraints,
                   cost_per_cluster):
    # pull out all of the points
    points = np.vstack([x.transformed_rep for x in leaf_nodes])
    num_points = points.shape[0]
    num_constraints = len(constraints)

    # run clustering and produce the linkage matrix
    Z = custom_hac(points, sim_func)

    # build the tree
    pred_tree_nodes = copy.copy(leaf_nodes) # shallow copy on purpose!
    new_node_id = num_points
    assert new_node_id == len(pred_tree_nodes)
    for merge_idx, merger in enumerate(Z):
        lchild, rchild = int(merger[0]), int(merger[1])
        lc_tr = pred_tree_nodes[lchild].transformed_rep
        rc_tr = pred_tree_nodes[rchild].transformed_rep
        
        default_conflict_tr = np.ones_like(lc_tr) * -np.inf

        # if invalid merger
        if np.array_equal(default_conflict_tr, lc_tr)\
                or np.array_equal(default_conflict_tr, rc_tr)\
                or np.any(np.abs(lc_tr - rc_tr) == 2):
            new_transformed_rep = default_conflict_tr
        else:
            new_transformed_rep = lc_tr | rc_tr

        pred_tree_nodes.append(
            TreeNode(
                new_node_id,
                pred_tree_nodes[lchild].raw_rep | pred_tree_nodes[rchild].raw_rep,
                transformed_rep=new_transformed_rep,
                children=[pred_tree_nodes[lchild], pred_tree_nodes[rchild]]
            )
        )
        new_node_id += 1

    # find the best cut
    cut_frontier_nodes, cut_obj_score = get_opt_tree_cut(
        pred_tree_nodes,
        sim_func,
        compat_func,
        constraints,
        cost_per_cluster,
        num_points,
        num_constraints
    )

    # the predicted entities canonicalization
    pred_canon_ents = np.vstack(
        [n.raw_rep for n in cut_frontier_nodes]
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
        'dp' : round(dp, 4),
        'adj_rand_idx' : round(adj_rand_idx, 4),
        'adj_mut_info' : round(adj_mut_info, 4),
        'cut_obj_score' : round(cut_obj_score, 4)
    }

    return pred_canon_ents, pred_labels, pred_tree_nodes, metrics


def gen_constraint(gold_entities,
                   pred_canon_ents,
                   pred_tree_nodes,
                   constraints,
                   sim_func,
                   num_to_generate=1):

    for _ in range(num_to_generate):
        # randomly generate valid feedback in the form of there-exists constraints
        pred_ent_idx = random.randint(0, len(pred_canon_ents)-1)
        ff_pred_ent = pred_canon_ents[pred_ent_idx]
        feat_intersect = (ff_pred_ent & gold_entities) == ff_pred_ent
        is_ent_subsets = np.all(feat_intersect, axis=1)
        num_ent_subsets = np.sum(is_ent_subsets)
        assert num_ent_subsets < 2
        if num_ent_subsets == 0:
            logger.debug('****** SPLIT CONSTRAINT ******')
            # split required
            tgt_ent_idx = np.argmax(np.sum(feat_intersect, axis=1))
            tgt_gold_ent = gold_entities[tgt_ent_idx]
            neg_match_feats = (feat_intersect[tgt_ent_idx] == False).astype(int)

            while True:
                ff_constraint = np.zeros_like(tgt_gold_ent)
                in_pred_domain = np.where(ff_pred_ent & tgt_gold_ent == 1)[0]
                out_pred_domain = np.where(neg_match_feats != 0)[0]
                in_idx = np.random.randint(in_pred_domain.size)
                out_idx = np.random.randint(out_pred_domain.size)
                ff_constraint[in_pred_domain[in_idx]] = 1
                ff_constraint[out_pred_domain[out_idx]] = -1

                if not any([np.array_equal(ff_constraint, xi) for xi in constraints]):
                    break
        else:
            logger.debug('****** MERGE CONSTRAINT ******')
            assert num_ent_subsets == 1
            # merge required
            super_gold_ent = 2*gold_entities[np.argmax(is_ent_subsets)]-1

            while True:
                ff_constraint = np.zeros_like(super_gold_ent)
                in_pred_domain = np.where(ff_pred_ent == 1)[0]
                out_pred_domain = np.where((super_gold_ent ^ ff_pred_ent) != 0)[0]
                in_idx = np.random.randint(in_pred_domain.size)
                out_idx = np.random.randint(out_pred_domain.size)
                ff_constraint[in_pred_domain[in_idx]] = 1
                ff_constraint[out_pred_domain[out_idx]] = super_gold_ent[out_pred_domain[out_idx]]

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
    constraints = []

    # construct tree node objects for leaves
    leaves = [TreeNode(i, m_rep) for i, m_rep in enumerate(mentions)]

    for r in range(opt.max_rounds):
        logger.debug('*** START - Clustering Points ***')
        # cluster the points
        out = cluster_points(
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


        logger.debug('*** START - Generating Constraints ***')
        # generate constraints and viable places given predictions
        constraints = gen_constraint(
            gold_entities,
            pred_canon_ents,
            pred_tree_nodes,
            constraints,
            sim_func,
            num_to_generate=opt.num_constraints_per_round
        )
        logger.debug('*** END - Generating Constraints ***')

        ## NOTE: JUST FOR TESTING
        #constraints = [(2*ent - 1) for ent in gold_entities]

        logger.debug('*** START - Computing Viable Placements ***')
        # update constraints and viable placements
        viable_placements = []
        for xi in constraints:
            compatible_nodes = constraint_compatible_nodes(
                pred_tree_nodes, xi, compat_func
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


def main():
    # get command line options
    opt = get_opt()

    # initialize the experiment
    initialize_exp(opt)

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

    # declare similarity function with function pointer
    sim_func = cos_sim
    compat_func = raw_overlap

    # run the core function
    run_mock_icff(
        opt, gold_entities, mentions, mention_labels, sim_func, compat_func
    )


if __name__ == '__main__':
    main()

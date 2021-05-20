import copy
import time
import pickle
import logging
from collections import deque 
from heapq import heapify, heappush, heappop

from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sparse_dot_mkl import dot_product_mkl

from assign import greedy_assign

from IPython import embed


if __name__ == '__main__':

    print('Loading test data...')
    with open('assign_test_input.pkl', 'rb') as f:
        pred_tree_nodes, constraints, viable_placements = pickle.load(f)
    print('Done.')

    placements_out = greedy_assign(
        pred_tree_nodes,
        constraints,
        viable_placements
    )

    embed()
    exit()



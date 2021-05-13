import os
import sys
import logging
import random
import pickle
import time
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from IPython import embed


MIN_FLOAT = np.finfo(float).min
MAX_FLOAT = np.finfo(float).max


def initialize_exp(opt):
    """
    Initialize the experiment:
    - create output directory (if not debug)
    - dump opt (if not debug)
    - create a logger
    - set the random seed
    """

    if not opt.debug:
        # create output directory
        assert not os.path.exists(opt.output_dir)
        os.makedirs(opt.output_dir)

        # dump opt
        pickle.dump(opt, open(os.path.join(opt.output_dir, 'opt.pkl'), "wb"))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            command.append("'%s'" % x)
    command = ' '.join(command)
    opt.command = command 

    # TODO: add git hash check if not in debug mode

    # create a logger
    logger = create_logger(opt)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(['%s: %s' % (k, str(v))
                           for k, v in sorted(dict(vars(opt)).items())]))
    if not opt.debug:
        logger.info('The experiment will be stored in %s\n' % opt.output_dir)
    logger.info('Running command: %s\n' % opt.command)

    # random seed
    set_seed(opt.seed)

    return logger


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(opt):
    """
    Create a logger.
    """

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create log formatter
    log_formatter = LogFormatter()

    # setup filepath for logger
    if not opt.debug:
        filepath = os.path.join(opt.output_dir, 'console.log')
        fh = logging.FileHandler(filepath, "a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)

    # create console handler and set level to info unless in debug mode
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if opt.debug else logging.INFO)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class InvalidAgglomError(Exception):
    pass


def sparse_agglom_rep(S):
    # sparse agglomeration of rows of S
    _S = S.tocoo()
    coord_map = defaultdict(set)
    for c, d in zip(_S.col, _S.data):
        coord_map[(0, c)].add(d)  # row is always 0
        
    if any([d == set([-1, 1]) for d in coord_map.values()]):
        raise InvalidAgglomError()

    agglom_row, agglom_col, agglom_data = zip(*[
        (r, c, list(d)[0]) for (r, c), d in coord_map.items()
    ])
    agglom_rep = csr_matrix(
        (agglom_data, (agglom_row, agglom_col)),
        shape=(1, S.shape[1]),
        dtype=float
    )
    return agglom_rep


def get_nil_rep(rep_dim=None):
    assert rep_dim is not None
    return csr_matrix(
        ([], ([], [])), shape=(1, rep_dim), dtype=float
    )

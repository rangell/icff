#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/ml_sl.py\
    --seed=27\
    --data_dir="data/real/"\
    --data_file="r8-test-stemmed.dataset.pkl"\
    --cost_per_cluster=1e-3\
    --max_rounds=100\
    --num_constraints_per_round=100\
    --sim_func='cosine'\
    --cluster_obj_reps='raw'\
    --debug

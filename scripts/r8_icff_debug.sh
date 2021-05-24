#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/icff.py\
    --seed=27\
    --data_dir="data/real/"\
    --data_file="r8-test-stemmed.dataset.pkl"\
    --cost_per_cluster=1e-3\
    --max_rounds=60\
    --num_constraints_per_round=20\
    --sim_func='cosine'\
    --compat_func='raw'\
    --constraint_strength=10\
    --cluster_obj_reps='raw'\
    --compat_agg='avg'\
    --debug

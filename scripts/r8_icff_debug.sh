#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/icff.py\
    --seed=27\
    --data_dir="data/real/"\
    --data_file="r8-test-stemmed.dataset.pkl"\
    --cost_per_cluster=7.5e-4\
    --max_rounds=60\
    --num_constraints_per_round=5\
    --sim_func='cosine'\
    --compat_func='raw'\
    --constraint_strength=5\
    --cluster_obj_reps='raw'\
    --compat_agg='avg'\
    --debug

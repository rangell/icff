#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/icff.py\
    --seed=27\
    --data_dir="data/real/"\
    --data_file="r52-test-stemmed.dataset.trimmed.pkl"\
    --cost_per_cluster=1e-3\
    --max_rounds=400\
    --num_constraints_per_round=5\
    --sim_func='cosine'\
    --compat_func='raw'\
    --constraint_strength=3\
    --cluster_obj_reps='raw'\
    --compat_agg='avg'\
    --debug

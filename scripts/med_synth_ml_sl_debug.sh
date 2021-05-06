#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/ml_sl.py\
    --seed=27\
    --data_dir="data/"\
    --num_entities=10\
    --num_mentions=100\
    --data_dim=256\
    --entity_noise_prob=0.4\
    --mention_sample_prob=0.35\
    --cost_per_cluster=0.005\
    --max_rounds=100\
    --num_constraints_per_round=1\
    --sim_func='cosine'\
    --cluster_obj_reps='raw'\
    --debug

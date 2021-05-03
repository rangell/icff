#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/main.py\
    --seed=27\
    --data_dir="data/"\
    --num_entities=2\
    --num_mentions=10\
    --data_dim=16\
    --max_rounds=50\
    --num_constraints_per_round=1\
    --entity_noise_prob=0.1\
    --mention_sample_prob=0.9\
    --cost_per_cluster=1.0\
    --debug

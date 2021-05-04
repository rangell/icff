#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/main.py\
    --seed=27\
    --data_dir="data/"\
    --num_entities=2\
    --num_mentions=10\
    --data_dim=16\
    --max_rounds=10\
    --num_constraints_per_round=1\
    --entity_noise_prob=0.2\
    --mention_sample_prob=0.6\
    --cost_per_cluster=0.1\
    --debug

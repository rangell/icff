#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/icff.py\
    --seed=27\
    --data_dir="data/synth/"\
    --num_entities=2\
    --num_mentions=10\
    --data_dim=16\
    --max_rounds=10\
    --num_constraints_per_round=1\
    --entity_noise_prob=0.3\
    --mention_sample_prob=0.4\
    --cost_per_cluster=0.1\
    --debug

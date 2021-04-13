#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/main.py\
    --seed=27\
    --data_dir="data/"\
    --num_entities=5\
    --num_mentions=25\
    --data_dim=16\
    --max_rounds=100\
    --num_constraints_per_round=1\
    --debug

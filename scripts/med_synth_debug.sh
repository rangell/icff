#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate icff

python src/main.py\
    --seed=27\
    --data_dir="data/"\
    --num_entities=10\
    --num_mentions=100\
    --data_dim=256\
    --max_rounds=50\
    --num_constraints_per_round=1\
    --debug

#!/bin/bash
#
#SBATCH --job-name=icff_{job_id}
#SBATCH --output=/mnt/nfs/scratch1/rangell/icff/experiments/logs/log_{job_id}.txt
#SBATCH -e /mnt/nfs/scratch1/rangell/icff/experiments/logs/res_{job_id}.err        # File to which STDERR will be written
#SBATCH --partition=longq
#
#SBATCH --ntasks=56
#SBATCH --time=0-12:00 
#SBATCH --mem-per-cpu=2000

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

eval "$(conda shell.bash hook)"
conda activate icff


export MKL_NUM_THREADS=56
export OPENBLAS_NUM_THREADS=56
export OMP_NUM_THREADS=56


pushd /mnt/nfs/scratch1/rangell/icff/

python src/icff.py\
    --seed={seed}\
    --data_dir="data/real/"\
    --data_file="{data_file}"\
    --output_dir="/mnt/nfs/scratch1/rangell/icff/experiments/exp{job_id}"\
    --cost_per_cluster=1e-3\
    --max_rounds=500\
    --num_constraints_per_round=5\
    --sim_func="cosine"\
    --compat_func="raw"\
    --constraint_strength=3\
    --cluster_obj_reps="raw"\
    --compat_agg="avg"

popd

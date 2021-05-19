#!/bin/bash
#
#SBATCH --job-name=cluster_linking_{job_id}
#SBATCH --output=/mnt/nfs/scratch1/rangell/icff/experiments/logs/log_{job_id}.txt
#SBATCH -e /mnt/nfs/scratch1/rangell/icff/experiments/logs/res_{job_id}.err        # File to which STDERR will be written
#SBATCH --partition=longq
#
#SBATCH --ntasks=56
#SBATCH --time=1-00:00 
#SBATCH --mem-per-cpu=2000

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

eval "$(conda shell.bash hook)"
conda activate icff

echo 'Hello World'

sleep 200
exit

export MKL_NUM_THREADS=56
export OPENBLAS_NUM_THREADS=56
export OMP_NUM_THREADS=56


pushd /mnt/nfs/scratch1/rangell/icff/

python src/icff.py\
    --seed=27\
    --data_dir="data/real/"\
    --data_file="r8-test-stemmed.dataset.pkl"\
    --output_dir="/mnt/nfs/scratch1/rangell/icff/experiments/exp{job_id}"\
    --cost_per_cluster={cost_per_cluster}\
    --max_rounds=50\
    --num_constraints_per_round={num_constraints_per_round}\
    --sim_func='cosine'\
    --compat_func='raw'\
    --constraint_strength={constraint_strength}\
    --cluster_obj_reps='raw'\
    --compat_agg={compat_agg}

popd

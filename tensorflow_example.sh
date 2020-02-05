#!/bin/bash
#SBATCH --time=60:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M

module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index tensorflow_gpu

python ./tensorflow_test.py


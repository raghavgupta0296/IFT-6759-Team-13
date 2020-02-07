#!/bin/bash
#SBATCH --time=60:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M

source ift6759-env/bin/activate

tensorboard --logdir logs --bind_all

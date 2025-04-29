#!/bin/bash

#SBATCH -A uli@a100
#SBATCH -C a100
#SBATCH --job-name=5_dev_a100_train_rgbd # nom du job
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8 # nbr de GPU par nœud
#SBATCH --cpus-per-task=6 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=2:00:00 # temps d'execution max
#SBATCH --output=log/5_dev_a100_train_rgbd%j.out # fichier sortie
#SBATCH --error=log/5_dev_a100_error_train_rgbd%j.out # fichier d'erreur

module load arch/a100

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

cd $WORK/$1
cd /lustre/fswork/projects/rech/tya/ubn15wo/$1
conda activate centergrasp_$1

#latest_ckpt=$(ls ckpt_sgdf/ | head -n 1) && jq --arg ckpt "$latest_ckpt" '.EmbeddingCkptPath = $ckpt' configs/rgb_train_specs.json > configs/rgb_train_specs.json.tmp && mv configs/rgb_train_specs.json.tmp configs/rgb_train_specs.json

export TMPDIR=$JOBSCRATCH
#WANDB_MODE=offline 

srun python scripts/train_rgbd.py --log-wandb





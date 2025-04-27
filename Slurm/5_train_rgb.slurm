#!/bin/bash

#SBATCH -A uli@v100
#SBATCH -C v100-32g
#SBvreveATCH --partition=gpu_p2
#SBATCH --job-name=5_train_rgbd # nom du job
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:3 # nbr de GPU par nœud
#SBATCH --cpus-per-task=30 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=100:00:00 # temps d'execution max
#SBATCH --output=log/5_train_rgbd%j.out # fichier sortie
#SBATCH --error=log/5_error_train_rgbd%j.out # fichier d'erreur
#SBgregregeATCH --array=6-9:3

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

cd $WORK/$1
conda activate centergrasp_$1

latest_ckpt=$(ls ckpt_sgdf/ | head -n 1) && jq --arg ckpt "$latest_ckpt" '.EmbeddingCkptPath = $ckpt' configs/rgb_train_specs.json > configs/rgb_train_specs.json.tmp && mv configs/rgb_train_specs.json.tmp configs/rgb_train_specs.json

export TMPDIR=$JOBSCRATCH
WANDB_MODE=offline python scripts/train_rgbd.py --log-wandb





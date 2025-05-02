#!/bin/bash

#SBATCH -A tya@v100
#SBATCH -C v100-32g
#SBAsdscTCH --partition=gpu_p2
#SBATCH --job-name=5_v100_train_rgbd # nom du job
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --exclusive
#SBcezcsATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4 # nbr de GPU par nœud
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=100:00:00 # temps d'execution max
#SBATCH --output=log/5_v100_train_rgbd%j.out # fichier sortie
#SBATCH --error=log/5_v100_error_train_rgbd%j.out # fichier d'erreur

module load arch/a100

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

cd $WORK/$1
cd /lustre/fswork/projects/rech/tya/ubn15wo/$1
conda activate centergrasp_$1

latest_ckpt=$(ls ckpt_sgdf/ | head -n 1) && jq --arg ckpt "$latest_ckpt" '.EmbeddingCkptPath = $ckpt' configs/rgb_train_specs.json > configs/rgb_train_specs.json.tmp && mv configs/rgb_train_specs.json.tmp configs/rgb_train_specs.json

export TMPDIR=$JOBSCRATCH
#WANDB_MODE=offline 

srun python scripts/train_rgbd.py --log-wandb





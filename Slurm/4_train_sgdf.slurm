#!/bin/bash

#SBATCH -A uli@v100
#SBATCH -C v100-32g
#SBAdezdTCH --partition=gpu_p2
#SBATCH --job-name=4_train_sgdf # nom du job
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/4_train_sgdf%j.out # fichier sortie
#SBATCH --error=log/4_error_train_sgdf%j.out # fichier d'erreur
#SBgregregeATCH --array=6-9:3

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

cd /lustre/fswork/projects/rech/tya/ubn15wo/$1
conda activate centergrasp_$1

export TMPDIR=$JOBSCRATCH
WANDB_MODE=offline python scripts/train_sgdf.py --log-wandb





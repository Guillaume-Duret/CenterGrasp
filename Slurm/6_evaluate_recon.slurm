#!/bin/bash

#SBATCH -A uli@v100
#SBAscsvTCH -C v100-32g
#SBATCH --partition=gpu_p2
#SBATCH --job-name=process # nom du job
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/log_gen_center%j.out # fichier sortie
#SBATCH --error=log/log_error_gen_center%j.out # fichier d'erreur
#SBgregregeATCH --array=6-9:3

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

cd $WORK/CenterGrasp
conda activate centergrasp_g_env

# evaluation of pretrained
python scripts/run_evals_shape.py

#python scripts/run_evals_grasp.py







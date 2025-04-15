#!/bin/bash

#SBATCH -A uli@v100
#SBATCH -C v100-32g
#SBATCH --job-name=process # nom du job
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=12:00:00 # temps d'execution max
#SBATCH --output=log/log_gen_center%j.out # fichier sortie
#SBATCH --error=log/log_error_gen_center%j.out # fichier d'erreur
#SBgregregeATCH --array=6-9:3

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

cd $WORK/CenterGrasp
conda activate centergrasp_g_env

#generate data
python scripts/make_grasp_labels.py --num-workers 10
python scripts/make_sgdf_dataset.py --num-workers 10
python scripts/make_rgb_dataset.py --headless --raytracing --num-workers 10 --mode train
python scripts/make_rgb_dataset.py --headless --raytracing --num-workers 10 --mode valid

# evaluation of pretrained
python scripts/run_evals_shape.py
python scripts/run_evals_grasp.py







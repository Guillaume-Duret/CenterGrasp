#!/bin/bash

#SBATCH -A uli@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --job-name=32_gen_rgb_val # nom du job
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBATCH --cpus-per-task=24 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/32_gen_rgb_center%j.out # fichier sortie
#SBATCH --error=log/32_error_gen_rgb_center%j.out # fichier d'erreur
#SBgregregeATCH --array=6-9:3


#module load pytorch-gpu/py3/1.13.0
#module unload cuda
#module load cuda/11.7.1

module load arch/h100
module load pytorch-gpu/py3/2.3.1

cd /lustre/fswork/projects/rech/tya/ubn15wo/$1
conda activate centergrasp_$1

#generate data
#python scripts/make_rgb_dataset.py --headless --raytracing --num-workers 15 --mode train
python scripts/make_rgb_dataset.py --headless --raytracing --num-workers 1 --mode valid







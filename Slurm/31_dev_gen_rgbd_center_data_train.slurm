#!/bin/bash

#SBATCH -A tya@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --job-name=31_dev_gen_rgb # nom du job
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:1 # nbr de GPU par nœud
#SBATCH --cpus-per-task=24 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=2:00:00 # temps d'execution max
#SBATCH --output=log/31_gen_rgb_center%j.out # fichier sortie
#SBATCH --error=log/31_error_gen_rgb_center%j.out # fichier d'erreur
#SBgregregeATCH --array=6-9:3


#module load pytorch-gpu/py3/1.13.0
#module unload cuda
#module load cuda/11.7.1

module load arch/h100
module load pytorch-gpu/py3/2.3.1

conda activate centergrasp_$1

cd /lustre/fswork/projects/rech/tya/ubn15wo/$1

#generate data
python scripts/make_rgb_dataset.py --headless --raytracing --num-workers 16 --mode train
#python scripts/make_rgb_dataset.py --headless --raytracing --num-workers 10 --mode valid







#!/bin/bash

#SBATCH -A uli@cpu
#SBATCH --job-name=giga_pile # nom du job
#SBATCH --qos=qos_cpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --cpus-per-task=40 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/log_process_obj%j.out # fichier sortie
#SBATCH --error=log/log_error_process_obj%j.out # fichier d'erreur
#SBAdsezfczeTCH --array=0-96:3

# nettoyage des modules charges en interactif et herites par defaut

module purge


module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1
conda activate centergrasp_g_env

cd /lustre/fswork/projects/rech/tya/ubn15wo/CenterGrasp/GIGA

python scripts/generate_data_parallel.py --scene pile --object-set pile/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/pile/40t3_no_data_pile_train_random_raw_4M






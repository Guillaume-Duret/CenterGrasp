#!/bin/bash

#SBATCH -A tya@cpu
#SBATCH --job-name=00_setup # nom du job
#SBATCH --qos=qos_cpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --cpus-per-task=10 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=2:00:00 # temps d'execution max
#SBATCH --output=log/00_setup%j.out # fichier sortie
#SBATCH --error=log/00_setup%j.out # fichier d'erreur
#SBAdsezfczeTCH --array=0-96:3

# nettoyage des modules charges en interactif et herites par defaut

module purge

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

conda create --name giga_$1 --clone process_giga --yes
conda activate giga_$1  # process_giga # centergrasp_g_env

cd /lustre/fswork/projects/rech/tya/ubn15wo/$1/GIGA_ori
#python -m pip install -e . --no-deps
pip install -e .

wait




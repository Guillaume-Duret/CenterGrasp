#!/bin/bash

#SBATCH -A tya@cpu
#SBATCH --job-name=1_giga_process # nom du job
#SBATCH --qos=qos_cpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --cpus-per-task=40 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/1_pro_giga%j.out # fichier sortie
#SBATCH --error=log/1_error_pro_giga%j.out # fichier d'erreur
#SBAdsezfczeTCH --array=0-96:3

# nettoyage des modules charges en interactif et herites par defaut

module purge

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

conda activate giga_$1 

#cd /lustre/fswork/projects/rech/tya/ubn15wo/CenterGrasp/GIGA
cd /lustre/fswork/projects/rech/tya/ubn15wo/$1/GIGA_ori/
#python scripts/generate_data_parallel.py --scene $1 --object-set $1/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/$1/$2

python scripts/clean_balance_data.py $2

python scripts/construct_dataset_parallel.py --num-proc 1 --single-view --add-noise dex $2 $3

python scripts/save_occ_data_parallel.py $2 100000 2 --num-proc 40











#!/bin/bash

#SBATCH -A uli@cpu
#SBATCH --job-name=giga_packed # nom du job
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
conda activate process_giga

cd /lustre/fswork/projects/rech/tya/ubn15wo/CenterGrasp/GIGA

#python scripts/generate_data_parallel.py --scene $1 --object-set $1/train --num-grasps 4000000 --num-proc 40 --save-scene ./data/$1/$2

python scripts/clean_balance_data.py $1

python scripts/construct_dataset_parallel.py --num-proc 1 --single-view --add-noise dex $1 $2

python scripts/save_occ_data_parallel.py $1 100000 2 --num-proc 40











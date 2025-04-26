#!/bin/bash

#SBATCH -A tya@cpu
#SBATCH --job-name=0_gen_data # nom du job
#SBATCH --qos=qos_cpu-t4
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --cpus-per-task=40 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=100:00:00 # temps d'execution max
#SBATCH --output=log/0_gen_data_obj%j.out # fichier sortie
#SBATCH --error=log/0_err_gen_data_obj%j.out # fichier d'erreur
#SBAdsezfczeTCH --array=0-96:3

# nettoyage des modules charges en interactif et herites par defaut

module purge

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

conda activate giga_$1

cd /lustre/fswork/projects/rech/tya/ubn15wo/$1/GIGA_ori
#pip install -e .

echo "in0"
pwd
echo $1

#python scripts/generate_data_parallel.py --scene $1 --object-set $1/train --num-grasps 4000 --num-proc 10 --save-scene ./data/$1/$2
python scripts/generate_data_parallel.py --scene $3 --object-set $3/train --num-grasps 5000000 --num-proc 40 --save-scene /lustre/fsn1/projects/rech/tya/ubn15wo/$2/data/$3/$4

wait




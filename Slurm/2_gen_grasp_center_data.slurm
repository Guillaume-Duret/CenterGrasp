#!/bin/bash

#SBATCH -A uli@v100
#SBfezfezATCH -C v100-32g
#SBATCH --job-name=2_gen_sgdf # nom du job
#SBATCH --qos=qos_gpu-t3
#SBATCH --ntasks=1 # nbr de tache MPI (= nbr de GPU)
#SBATCH --gres=gpu:4 # nbr de GPU par nœud
#SBATCH --cpus-per-task=40 # nbr de CPU par tache
#SBATCH --hint=nomultithread # pas de hyperthreading
#SBATCH --time=20:00:00 # temps d'execution max
#SBATCH --output=log/2_gen_sgdf%j.out # fichier sortie
#SBATCH --error=log/2_error_gen_sgdf%j.out # fichier d'erreur
#SBgregregeATCH --array=6-9:3

module load pytorch-gpu/py3/1.13.0
module unload cuda
module load cuda/11.7.1

#conda create --name centergrasp_$1 --clone centergrasp_g_env

cp -r $SCRATCH/.conda/envs/centergrasp_g_env $SCRATCH/.conda/envs/centergrasp_$1
conda activate centergrasp_$1  

cd /lustre/fswork/projects/rech/tya/ubn15wo/$1
python -m pip install -e . --no-deps

#generate data
python scripts/make_grasp_labels.py --num-workers 40
python scripts/make_sgdf_dataset.py --num-workers 40









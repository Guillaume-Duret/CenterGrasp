#!/bin/bash

# Check if exactly 4 arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <code_ref> <data_ref> <project_dir> <data_dir>"
    exit 1
fi

code_ref=$1
data_ref=$2
project_dir=$3
data_dir=$4

# Define the directory where your slurm scripts are located
SLURM_SCRIPT_DIR="/lustre/fswork/projects/rech/tya/ubn15wo/$project_dir/Slurm"  # CHANGE THIS TO YOUR ACTUAL PATH

# Copy directories
#cp -rf $WORK/CenterGrasp_ref_dev $WORK/$project_dir
#cp -rf $SCRATCH/Centergrasp_data_ref $SCRATCH/$data_dir

cp -rf $WORK/$code_ref $WORK/$project_dir
cp -rf $SCRATCH/$data_ref $SCRATCH/$data_dir

ln -sf $SCRATCH/$data_dir $WORK/$project_dir/Centergrasp_data_Test

# Create wandb directory and symlink
cd $WORK/$project_dir
mkdir -p /lustre/fsn1/projects/rech/tya/ubn15wo/$data_dir/wandb
ln -sf /lustre/fsn1/projects/rech/tya/ubn15wo/$data_dir/wandb wandb

cd $WORK/$project_dir/Slurm
# Submit jobs with dependencies
# First submit the independent jobs (0_gen_giga_data)

echo "0"
pwd

job0=$(sbatch --parsable ${SLURM_SCRIPT_DIR}/0_setup.slurm $project_dir)

job1=$(sbatch --parsable --dependency=afterok:$job0 ${SLURM_SCRIPT_DIR}/0_gen_giga_data.slurm $project_dir $data_dir pile gen_data_pile_train_random_raw_4M)
job2=$(sbatch --parsable --dependency=afterok:$job0 ${SLURM_SCRIPT_DIR}/0_gen_giga_data.slurm $project_dir $data_dir packed gen_data_packed_train_random_raw_4M)

echo "1"

# Process data depends on both generation jobs
job3=$(sbatch --parsable --dependency=afterok:$job1:$job2 ${SLURM_SCRIPT_DIR}/1_process_giga_data.slurm $project_dir $SCRATCH/$data_dir/data/pile/gen_data_pile_train_random_raw_4M/ $SCRATCH/$data_dir/data/pile/gen_data_pile_train_random_4M_noise/)
job4=$(sbatch --parsable --dependency=afterok:$job1:$job2 ${SLURM_SCRIPT_DIR}/1_process_giga_data.slurm $project_dir $SCRATCH/$data_dir/data/packed/gen_data_packed_train_random_raw_4M/ $SCRATCH/$data_dir/data/packed/gen_data_packed_train_random_4M_noise/)

# Grasp center data depends on both processing jobs
job5=$(sbatch --parsable --dependency=afterok:$job3:$job4 ${SLURM_SCRIPT_DIR}/2_gen_grasp_center_data.slurm $project_dir)

# RGBD generation depends on grasp center data
job6=$(sbatch --parsable --dependency=afterok:$job5 ${SLURM_SCRIPT_DIR}/31_gen_rgbd_center_data_train.slurm $project_dir)
job7=$(sbatch --parsable --dependency=afterok:$job5 ${SLURM_SCRIPT_DIR}/32_gen_rgbd_center_data_val.slurm $project_dir)

# Training jobs depend on RGBD generation
job8=$(sbatch --parsable --dependency=afterok:$job5 ${SLURM_SCRIPT_DIR}/4_train_sgdf.slurm $project_dir)
job9=$(sbatch --parsable --dependency=afterok:$job8:$job6:$job7 ${SLURM_SCRIPT_DIR}/5_train_rgb.slurm $project_dir)

echo "All jobs submitted with dependencies:"
echo "Data generation jobs: $job1, $job2"
echo "Data processing jobs: $job3, $job4"
echo "Grasp center generation: $job5"
echo "RGBD generation: $job6, $job7"
echo "Training jobs: $job8, $job9"

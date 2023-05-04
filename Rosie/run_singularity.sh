#!/bin/bash

#SBATCH --job-name='batch_job'
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=8
#SBATCH --time=0-5:0
# time format: <days>-<hours>:<minutes>

# Load Singularity module
module load singularity

# Run the script inside the Singularity container
singularity exec --nv /home/eckbergj/E2EFT/Singularity.sif python3 /home/eckbergj/E2EFT/my_script.py




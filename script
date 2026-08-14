#!/bin/bash
# Usage: sbatch slurm-gpu-job-script
# Prepared By: Kai Xi,  Feb 2015
#              help@massive.org.au

# Modified By : Isaac Ning Lee,  Dec 2023

# NOTE: To activate a SLURM option, remove the whitespace between the '#' and 'SBATCH'

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=vits

# Request CPU resource for a serial job
# SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# SBATCH --exclude=mlerp-monash-node05,mlerp-monash-node06

# Request for GPU, 
#
# Option 1: Choose any GPU whatever m2070 or K20
# Note in most cases, 'gpu:N' should match '--ntasks=N'
#SBATCH --gres=gpu:40gb:1
#SBATCH --partition=BigCats
#SBATCH --qos=Lion

#SBATCH --time=1-00:00:00

#SBATCH --mail-user=ilee0022@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=/home/ilee0022/cl-gen/calcury/vits/logs/benchmark/train.out
#SBATCH --error=/home/ilee0022/cl-gen/calcury/vits/logs/benchmark/train.err

nvidia-smi
source /mnt/userdata3/ilee0022/cl-gen/miniforge3/bin/activate
conda activate vits-env

# espeak-ng env
export PATH="/home/ilee0022/cl-gen/calcury/espeak-ng-install/bin:$PATH"
export ESPEAK_DATA_PATH="/home/ilee0022/cl-gen/calcury/espeak-ng-install/share/espeak-ng-data"

cd /home/ilee0022/cl-gen/calcury/vits

python train_ms.py -c configs/vctk_base.json -m vctk_base
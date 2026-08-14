#!/bin/bash
# SalUn speaker-unlearning fine-tune (very low LR) on the MASSIVE/Monash cluster.
# Prepared By: Kai Xi, Feb 2015 / Isaac Ning Lee, Dec 2023 (template)
# Usage: sbatch slurm_finetune.sh

#SBATCH --job-name=vits-salun

# Request CPU resource for a serial job
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# SBATCH --exclude=mlerp-monash-node05,mlerp-monash-node06

# Request for GPU
#SBATCH --gres=gpu:40gb:1
#SBATCH --partition=BigCats
#SBATCH --qos=Lion

#SBATCH --time=1-00:00:00

#SBATCH --mail-user=ilee0022@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/ilee0022/cl-gen/calcury/vits/logs/finetune/train.out
#SBATCH --error=/home/ilee0022/cl-gen/calcury/vits/logs/finetune/train.err

mkdir -p /home/ilee0022/cl-gen/calcury/vits/logs/finetune
nvidia-smi
source /mnt/userdata3/ilee0022/cl-gen/miniforge3/bin/activate
conda activate vits-env

# espeak-ng env (text cleaning; not strictly needed for cleaned filelists)
export PATH="/home/ilee0022/cl-gen/calcury/espeak-ng-install/bin:$PATH"
export ESPEAK_DATA_PATH="/home/ilee0022/cl-gen/calcury/espeak-ng-install/share/espeak-ng-data"

cd /home/ilee0022/cl-gen/calcury/vits

# SalUn: load pretrained_vctk.pth, compute a weight-saliency mask on the forget
# speaker's data, then fine-tune only the salient weights at a very low LR.
# Checkpoints (G_unlearn_*.pth) are saved every --save-every epochs and can be
# fed to `unlearning_evaluatioin.py --unlearned <ckpt>`.
python train_unlearn_salun.py \
    -c configs/vctk_base.json \
    -m vctk_unlearn_salun \
    --pretrained pretrained/pretrained_vctk.pth \
    --forget-speaker p231 \
    --lr 1e-6 \
    --epochs 10 \
    --save-every 2 \
    --sparsity 0.1

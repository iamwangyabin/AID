#!/bin/bash

##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --partition=a100
##SBATCH --account=ecsstaff
##SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:1
##SBATCH --time=60:00:00

#eval "$(conda shell.bash hook)"
#conda init bash
#conda activate cl

#export HF_HOME=/scratch/yw26g23/cache/
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE="offline"
export WANDB_API_KEY="a4d3a740e939973b02ac59fbd8ed0d6a151df34b"
export NO_ALBUMENTATIONS_UPDATE=1

python train.py --cfg cfgs/train/fakecoco/ojha_fcoco_sd3.yaml












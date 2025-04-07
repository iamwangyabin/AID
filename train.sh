#!/bin/bash


#conda activate cl

#export HF_HOME=/scratch/yw26g23/cache/
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE="offline"
export WANDB_API_KEY="a4d3a740e939973b02ac59fbd8ed0d6a151df34b"
export NO_ALBUMENTATIONS_UPDATE=1

python train.py --cfg cfgs/train/official/rine_official.yaml


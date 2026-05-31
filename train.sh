#!/bin/bash


#conda activate cl

#export HF_HOME=/scratch/yw26g23/cache/
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="offline"
export NO_ALBUMENTATIONS_UPDATE=1

python train.py --cfg "${AID_TRAIN_CFG:-cfgs/train/official/rine_official.yaml}"

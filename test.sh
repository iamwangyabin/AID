#!/bin/bash

export WANDB_MODE="offline"
export NO_ALBUMENTATIONS_UPDATE=1

python test.py --cfg "${AID_TEST_CFG:-cfgs/test/official/rine_official.yaml}"

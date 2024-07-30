#!/bin/bash
export PYTHONPATH=/workspace/DIST_KD/classification
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/strategies/distill/resnet_kdt4.yaml --model tv_resnet18 --experiment KD_LS10_t4

#!/bin/bash

# 3 fold training
CUDA_VISIBLE_DEVICES=0 python train_patch_level.py --k 0 --model_name efficientnet --exp train_TCGA_CRC_Kather_3fold --splits data_info/TCGA_CRC_Kather_3fold.csv
CUDA_VISIBLE_DEVICES=0 python train_patch_level.py --k 1 --model_name efficientnet --exp train_TCGA_CRC_Kather_3fold --splits data_info/TCGA_CRC_Kather_3fold.csv
CUDA_VISIBLE_DEVICES=0 python train_patch_level.py --k 2 --model_name efficientnet --exp train_TCGA_CRC_Kather_3fold --splits data_info/TCGA_CRC_Kather_3fold.csv
#!/bin/bash

MODEL_TYPE=${1}     # llama
MODEL_SIZE=${2}     # 13b
DEVICE=${3}         # 0

CUDA_VISIBLE_DEVICES=${DEVICE} \
python ./intactkv_motivation.py \
    --dataset_path ./datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --max_train_samples 128 --max_seq_len 1024 \
    --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
    --quant_method rtn --bits 3 --group_size 128 \
    --intactkv_size 34

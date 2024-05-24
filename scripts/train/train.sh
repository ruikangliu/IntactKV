#!/bin/bash

MODEL_TYPE=${1}     # vicuna-v1.5
MODEL_SIZE=${2}     # 7b
QUANT_METHOD=${3}   # awq
BITS=${4}           # 3
DEVICE=${5}         # 0

set -v

# intactkv
CUDA_VISIBLE_DEVICES=${DEVICE} \
python ./intactkv_train.py \
    --dataset_path ./datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --max_train_samples 128 \
    --learning_rate 2e-4 \
    --max_steps 160 \
    --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
    --quant_method ${QUANT_METHOD} --bits ${BITS} --group_size 128

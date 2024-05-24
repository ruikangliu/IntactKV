#!/bin/bash

MODEL_TYPE=${1} # vicuna-v1.5
MODEL_NAME=${2} # vicuna-v1.5-7b
BITS=${3}       # 3
GROUP_SIZE=${4} # 128
DEVICE=${5}     # 0

set -v

CUDA_VISIBLE_DEVICES=${DEVICE} \
python quantize_with_gptq.py \
    --pretrained_model_dir ./modelzoo/${MODEL_TYPE}/${MODEL_NAME} \
    --quantized_model_dir ./modelzoo/autogptq/${MODEL_TYPE}/${MODEL_NAME}-${BITS}bit-${GROUP_SIZE}g \
    --bits ${BITS} \
    --group_size ${GROUP_SIZE} \
    --desc_act \
    --num_samples 128 \
    --save_and_reload

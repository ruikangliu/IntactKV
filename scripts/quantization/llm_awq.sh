#!/bin/bash

MODEL_TYPE=${1} # vicuna-v1.5
MODEL_SIZE=${2} # 7b
BITS=${3}       # 3
GROUP_SIZE=${4} # 128
DEVICE=${5}     # 0

set -v

CUDA_VISIBLE_DEVICES=${DEVICE} \
python quantize_with_awq.py \
    --run_awq \
    --model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
    --q_group_size ${GROUP_SIZE} \
    --w_bit ${BITS} \
    --dump_awq ./modelzoo/llm-awq/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE}-w${BITS}-g${GROUP_SIZE}.pt

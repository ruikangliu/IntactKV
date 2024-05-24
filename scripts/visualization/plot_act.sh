#!/bin/bash

MODEL_TYPE=${1}     # vicuna-v1.5
MODEL_SIZE=${2}     # 7b
DEVICE=${3}         # 0

CUDA_VISIBLE_DEVICES=${DEVICE} \
python ./plot_activations.py \
    --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE}

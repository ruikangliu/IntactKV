#!/bin/bash

MODEL_TYPE=${1}     # vicuna-v1.5
MODEL_SIZE=${2}     # 7b
QUANT_METHOD=${3}   # awq
BITS=${4}           # 3
KV_BITS=${5}           # 4
TASK=${6}           # ppl/mmlu/qa
PORT=${7}           # 29500
DEVICE=${8}         # 0

set -v

if [ ${TASK} == "ppl" ];
then
    # vanilla
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --tasks ${TASK}

    # intactkv
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --intactkv \
        --tasks ${TASK}
else
    # vanilla
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    accelerate launch --main_process_port ${PORT} ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --tasks ${TASK}

    # intactkv
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    accelerate launch --main_process_port ${PORT} ./intactkv_eval.py \
        --fp16_model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} \
        --bits ${BITS} --kv_bits ${KV_BITS} --group_size 128 \
        --intactkv \
        --tasks ${TASK}
fi

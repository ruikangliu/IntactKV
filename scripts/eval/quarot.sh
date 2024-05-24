#!/bin/bash

MODEL_TYPE=${1}     # llama-2
MODEL_SIZE=${2}     # 7b
DEVICE=${3}         # 0

set -v

# quarot
CUDA_VISIBLE_DEVICES=${DEVICE} \
python ./intactkv_quarot.py \
    --model ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
    --rotate \
    --act_order --w_bits 4 --w_clip \
    --a_bits 4 --a_clip_ratio 0.9 \
    --k_bits 4 --k_asym --k_groupsize 128 --k_clip_ratio 0.95 \
    --v_bits 4 --v_asym --v_groupsize 128 --v_clip_ratio 0.95

# quarot+IntactKV
CUDA_VISIBLE_DEVICES=${DEVICE} \
python ./intactkv_quarot.py \
    --model ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
    --rotate \
    --act_order --w_bits 4 --w_clip \
    --a_bits 4 --a_clip_ratio 0.9 \
    --k_bits 4 --k_asym --k_groupsize 128 --k_clip_ratio 0.95 \
    --v_bits 4 --v_asym --v_groupsize 128 --v_clip_ratio 0.95 \
    --intactkv

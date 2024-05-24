#!/bin/bash

MODEL_TYPE=${1} # vicuna-v1.5
MODEL_SIZE=${2} # 7b
QUANT_METHOD=${3}   # awq
BITS=${4}       # 3
DEVICE=${5}     # 0

set -v

for SEED in {42..44};
do
    # vanilla
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./gen_mtbench_answer.py \
        --model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} --bits ${BITS} --group_size 128 \
        --seed ${SEED}
    
    # IntactKV + Cal
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python ./gen_mtbench_answer.py \
        --model_path ./modelzoo/${MODEL_TYPE}/${MODEL_TYPE}-${MODEL_SIZE} \
        --quant_method ${QUANT_METHOD} --bits ${BITS} --group_size 128 \
        --intactkv --intactkv_path intactkv/epoch_20 \
        --seed ${SEED}
done

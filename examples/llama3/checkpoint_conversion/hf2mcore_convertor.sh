#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=7
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_SIZE=$1
HG_CKPT_PATH=$2
MEGATRON_PATH=$3
SOURCE_CKPT_PATH=$4
TARGET_CKPT_PATH=$5
TP=$6
PP=$7
EXTRA_VOCAB_SIZE=$8
mg2hf=${9}

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=28672
NUM_KV_HEADS=8
VOCAB_SIZE=128256
ROPE_THETA=500000


if [ "$mg2hf" = true ]; then
    convert_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
else
    convert_options=""
fi

python hf2mcore_70b.py \
  --load ${HG_CKPT_PATH} \
  --load_path ${SOURCE_CKPT_PATH} \
  --save_path ${TARGET_CKPT_PATH} \
  --target_params_dtype bf16 \
  --target_tensor_model_parallel_size ${TP} \
  --target_pipeline_model_parallel_size ${PP} \
${convert_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"

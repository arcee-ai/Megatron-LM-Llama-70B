#!/bin/bash

# Arguments
MODEL_SIZE="70B"                                                                          # Model size: 7B, 13B, 70B
HG_CKPT_PATH="/workspace/1.megatron-lm-files/huggingface-files/hf_models/llama3_70B_base/models--meta-llama--Meta-Llama-3-70B/snapshots/b4d08b7db49d488da3ac49adf25a6b9ac01ae338"                      # Path to HF checkpoint
MEGATRON_PATH="./"                                                                       # Path to 1.megatron-lm-files root directory
SOURCE_CKPT_PATH="/workspace/1.megatron-lm-files/huggingface-files/hf_models/llama3_70B_base/models--meta-llama--Meta-Llama-3-70B/snapshots/b4d08b7db49d488da3ac49adf25a6b9ac01ae338"
TARGET_CKPT_PATH="/workspace/1.megatron-lm-files/huggingface-files/mg_converted_TP8_PP4"  # Target path
TP=8                                                                                      # Tensor Parallelism
PP=4                                                                                     # Pipeline Parallelism
EXTRA_VOCAB_SIZE=256                                                                      # Extra vocabulary size
mg2hf="false"                                                                             # Whether to execute mcore2hf conversion

# Run the conversion script with provided arguments
bash hf2mcore_convertor.sh \
    $MODEL_SIZE \
    $HG_CKPT_PATH \
    $MEGATRON_PATH \
    $SOURCE_CKPT_PATH \
    $TARGET_CKPT_PATH \
    $TP \
    $PP \
    $EXTRA_VOCAB_SIZE \
    $NUM_EXPERTS \
    $EXPERTS_TOPK \
    $EP \
    $NUM_EXPERT_SPLITS \
    $mg2hf

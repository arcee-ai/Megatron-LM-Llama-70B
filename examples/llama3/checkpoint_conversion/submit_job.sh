#!/bin/bash

# Arguments
MODEL_SIZE="70B"                                                                          # Model size: 7B, 13B, 70B
HG_CKPT_PATH="/workspace/1.megatron-lm/checkpoint-conversion/Llama3-70B-Checkpoint/Meta-Llama-3-70B-Instruct"                      # Path to HF checkpoint
MEGATRON_PATH="../../../"                                                                       # Path to Megatron-LM root directory
#SOURCE_CKPT_PATH="/workspace/llama3-70B-checkpoints/Meta-Llama-3-70B-to-mcore-tp8-pp4/release"  # Source path
SOURCE_CKPT_PATH="/workspace/1.megatron-lm/hyperpod_checkpoints/iter_0000953-INSTRUCT"
TARGET_CKPT_PATH="/workspace/1.megatron-lm/hyperpod_checkpoints/instruct-trained"                             # Target path
TP=8                                                                                      # Tensor Parallelism
PP=4                                                                                      # Pipeline Parallelism
EXTRA_VOCAB_SIZE=256                                                                      # Extra vocabulary size
NUM_EXPERTS=0                                                                             # Number of experts
EXPERTS_TOPK=0                                                                            # Topk for expert routing
EP=0                                                                                      # Expert parallelism
NUM_EXPERT_SPLITS=0                                                                          
mg2hf="true"                                                                             # Whether to execute mcore2hf conversion

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

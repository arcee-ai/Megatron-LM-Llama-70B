#!/bin/bash

# HARDCODED VARIABLES
MEGATRON_PATCH_PATH=../..                                                           # SET THE PATH OF THE MEGATRON PATCH CODE
INPUT_DATA_DIR=/workspace/llama3-70B-checkpoints/raw_data/oscar-1GB.jsonl           # PATH OF THE PACKAGED WUDAO DATASET FOLDER
TOKENIZER=llamabpe                                                                  # TOKENIZER TYPE
OUTPUT_DATA_DIR=/workspace/llama3-70B-checkpoints/llama3_data                       # DIRECTORY FOR OUTPUT BIN AND IDX FILES
HF_TOKENIZER_PATH=/workspace/llama3-70B-checkpoints/Meta-Llama-3-70B-Instruct       # PATH OF THE TOKENIZER_CONFIG.JSON FILE

# RUN THE PRETRAINING DATASET CREATION SCRIPT
bash run_make_pretraining_dataset_megatron.sh \
$MEGATRON_PATCH_PATH \
$INPUT_DATA_DIR \
$TOKENIZER \
$OUTPUT_DATA_DIR \
$HF_TOKENIZER_PATH
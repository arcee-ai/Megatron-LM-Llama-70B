#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --nodes=4 # number of nodes to use, 2 p4d(e) = 16 A100 GPUs
#SBATCH --job-name=megatron_gpt # name of your job
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1

set -ex;

###########################
###### User Variables #####
###########################

# Parallelism decomposition variables
: "${TENSOR_PARALLEL:=4}"
: "${PIPELINE_PARALLEL:=1}"


# Model parameters, defaults to 39B model
# Refer to page 8 of this paper on how to tune models parameters
# https://arxiv.org/pdf/2104.04473.pdf
: "${NUM_LAYERS:=28}"
: "${HIDDEN_SIZE:=3584}"
: "${NUM_ATTENTION_HEADS:=28}"

: "${SEQ_LENGTH:=16384}"
: "${MAX_POSITION_EMBEDDINGS:=131072}"
: "${MICRO_BATCH_SIZE:=2}"
: "${GLOBAL_BATCH_SIZE:=128}"

: "${EXTRA_VOCAB_SIZE:=421}"

# default variables for Enroot
: "${IMAGE:=$(pwd)/qwen-training.sqsh}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$(pwd):$DATA_PATH}"

###########################
## Environment Variables ##
###########################

# https://discuss.pytorch.org/t/nccl-network-is-unreachable-connection-refused-when-initializing-ddp/137352
# https://github.com/pytorch/pytorch/issues/68893
# export NCCL_SOCKET_IFNAME=ens
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO

# async runtime error ...
export CUDA_DEVICE_MAX_CONNECTIONS=1

# # weights and biases API-KEY
export WANDB_API_KEY="fab4c867547561879ec227778fd6fdb04fc5420a"

#########################
## Command and Options ##
#########################

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $FSX_MOUNT
)

declare -a TORCHRUN_ARGS=(
    # change this to match the number of gpus per node:
    --nproc_per_node=8
    --nnodes=$SLURM_JOB_NUM_NODES
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$(hostname)
)

##########################
### Get Train iters
##########################
# Training parameters with default values if not set
: "${TRAIN_TOKENS:=80118284333}"
: "${WARMUP_TOKENS:=8011828433}"

# Compute training iterations based on tokens, global batch size, and sequence length
: "${TRAIN_ITERS:=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))}"
: "${LR_WARMUP_ITERS:=$(( ${WARMUP_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))}"
: "${LR_DECAY_ITERS:=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LENGTH} ))}"



declare -a MODEL_ARGS=(
        --use-mcore-models
        --disable-bias-linear
        --seq-length $SEQ_LENGTH
        --max-position-embeddings $MAX_POSITION_EMBEDDINGS
        --num-layers $NUM_LAYERS
        --hidden-size $HIDDEN_SIZE
        --ffn-hidden-size 18944
        --num-attention-heads $NUM_ATTENTION_HEADS
        --init-method-std 0.008
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --normalization RMSNorm
        --norm-epsilon 1e-06
        --position-embedding-type rope
        --rotary-base 1000000
        --rotary-seq-len-interpolation-factor 1 
        --use-rotary-position-embeddings
        --swiglu
        --untie-embeddings-and-output-weights
        --group-query-attention
        --num-query-groups 4
        --no-masked-softmax-fusion
        --add-qkv-bias
)

declare -a TRAINING_ARGS=(
        --micro-batch-size $MICRO_BATCH_SIZE
        --global-batch-size $GLOBAL_BATCH_SIZE
        --lr 1.0e-5
        --train-iters $TRAIN_ITERS
        --eval-iters 25 
        --eval-interval 1000 
        --lr-decay-iters $LR_DECAY_ITERS
        --lr-decay-style cosine
        --lr-warmup-iters $LR_WARMUP_ITERS
        --split 99,1,0 
        --min-lr 1.0e-6
        --weight-decay 0.1
        --adam-beta1 0.9
        --adam-beta2 0.95 
        --clip-grad 1.0
        --bf16
        --overlap-grad-reduce
        --overlap-param-gather
        --use-flash-attn
        --spectrum_freeze
)


declare -a MEGATRON_PARALLELISM=(
        --tensor-model-parallel-size $TENSOR_PARALLEL
        --pipeline-model-parallel-size $PIPELINE_PARALLEL
        --sequence-parallel
        --use-distributed-optimizer
)

declare -a CHECKPOINTING_ARGS=(
        --save "${DATA_PATH}/hyperpod_checkpoints" 
        --load "${DATA_PATH}/hf_checkpoints/qwen2_7b_instruct/megatron_4tp/" 
        --no-load-optim
        --no-save-optim
        --no-load-rng
        --no-save-rng
        --exit-on-missing-checkpoint
        --save-interval 10000
)

declare -a DATA_ARGS=(
        #--data-path "${DATA_PATH}/data/qwen2-instrcut-mg-data/qwen-2-instruct_text_document" 
        --data-path "${DATA_PATH}/data/qwen2-SEC-data/qwen2-instruct-sec-tokenized_text_document" 
        #--data-path "${DATA_PATH}/data/qwen-datasets/wudao_qwenbpe_text_document"
        --hf-tokenizer-path "${DATA_PATH}/data/qwen2-SEC-data/tokenizer"
        #--hf-tokenizer-path "Qwen/Qwen2-7B-Instruct"
        --tokenizer-type Qwen2Tokenizer
        --extra-hf-tokens $EXTRA_VOCAB_SIZE
)

declare -a LOGGING_ARGS=(
        --wandb-project ${WANDB_PROJECT:-"Hyperpod-Mixtral-Qwen2-7B"}
        --wandb-exp-name ${WANDB_NAME:-"Qwen2-7B-instruct-sec80B-TE"}
        --tensorboard-dir "${DATA_PATH}/hyperpod_checkpoints/tensorboard-cpt" 
        --log-interval 1 
        --log-progress
)

AUTO_RESUME=""
if [ -d "/opt/sagemaker_cluster" ]; then
    echo "Detected Hyperpod cluster.. enabling --auto-resume=1"
    AUTO_RESUME="--auto-resume=1"
fi



# Command to run the training script with distributed settings
srun    ${AUTO_RESUME} -l "${ARGS[@]}" python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" pretrain_gpt.py \
        "${MEGATRON_PARALLELISM[@]}" \
        "${MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${CHECKPOINTING_ARGS[@]}" \
        "${LOGGING_ARGS[@]}"
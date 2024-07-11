
set -ex;

###########################
###### User Variables #####
###########################

# Parallelism decomposition variables
: "${TENSOR_PARALLEL:=1}"
: "${PIPELINE_PARALLEL:=1}"


# Model parameters, defaults to 39B model
# Refer to page 8 of this paper on how to tune models parameters
# https://arxiv.org/pdf/2104.04473.pdf
: "${NUM_LAYERS:=1}"
: "${HIDDEN_SIZE:=8192}"
: "${NUM_ATTENTION_HEADS:=64}"

: "${SEQ_LENGTH:=8192}"
: "${MAX_POSITION_EMBEDDINGS:=8192}"
: "${MICRO_BATCH_SIZE:=1}"
: "${GLOBAL_BATCH_SIZE:=8}"

: "${EXTRA_VOCAB_SIZE:=256}"

export CUDA_DEVICE_MAX_CONNECTIONS=1


###########################
## Environment Variables ##
###########################


declare -a TORCHRUN_ARGS=(
    # change this to match the number of gpus per node:
    --nproc_per_node=8
    --nnodes=1
    --master_addr=localhost
    --master_port=7000
)

##########################
### Get Train iters
##########################
# Training parameters with default values if not set
: "${TRAIN_TOKENS:=100000000}"
: "${WARMUP_TOKENS:=10000}"

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
        --ffn-hidden-size 28672
        --num-attention-heads $NUM_ATTENTION_HEADS
        --init-method-std 0.01
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --normalization RMSNorm
        --norm-epsilon 1e-05 
        --position-embedding-type rope
        --rotary-base 500000
        # --no-rope-fusion 
        --use-rotary-position-embeddings
        --swiglu
        --untie-embeddings-and-output-weights
        --group-query-attention
        --num-query-groups 8
        # --no-masked-softmax-fusion
)

declare -a TRAINING_ARGS=(
        --micro-batch-size $MICRO_BATCH_SIZE
        --global-batch-size $GLOBAL_BATCH_SIZE
        --lr 1.0e-5
        --train-iters $TRAIN_ITERS
        --eval-iters 40 
        --eval-interval 10000 
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
        #--use-flash-attn
        # --make-vocab-size-divisible-by 64
)


declare -a MEGATRON_PARALLELISM=(
        --tensor-model-parallel-size $TENSOR_PARALLEL
        --pipeline-model-parallel-size $PIPELINE_PARALLEL
        --sequence-parallel
        --use-distributed-optimizer
)

declare -a DATA_ARGS=(
        --data-path "/workspace/Luke-Shamane-processing/tokenized_data/shamane-luke_text_document" 
        --hf-tokenizer-path "/workspace/Luke-Shamane-processing/tokenizer/base-llama3-70B"
        --tokenizer-type Llama3Tokenizer
        --extra-hf-tokens $EXTRA_VOCAB_SIZE

)

declare -a LOGGING_ARGS=(
        --log-interval 1 
        --log-progress
)


# Command to run the training script with distributed settings
torchrun "${TORCHRUN_ARGS[@]}" pretrain_llama3_70b.py \
        "${MEGATRON_PARALLELISM[@]}" \
        "${MODEL_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${LOGGING_ARGS[@]}"
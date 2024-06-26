#! /bin/bash
START_TIME=$SECONDS
MEGATRON_PATCH_PATH=$1
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240519
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
input_data_dir=$2
tokenizer=$3
output_data_dir=$4
load_dir=$5

INPUT="${input_data_dir}"


if [ $tokenizer = "llamabpe" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/SlimPajama_llamabpe \
  --patch-tokenizer-type Llama3Tokenizer \
  --tokenizer-type GPT2BPETokenizer \
  --hf-tokenizer-path ${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"

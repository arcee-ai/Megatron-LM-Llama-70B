python3 tools/preprocess_data.py \
        --input /workspace/megatron-lm/SEC-DATA/final_output.jsonl \
        --output-prefix   /workspace/megatron-lm/SEC-DATA/sec-tokenized \
        --hf-tokenizer-path /workspace/megatron-lm/huggingface-files/tokenizers/llama3_70B_base/models--meta-llama--Meta-Llama-3-70B/snapshots/b4d08b7db49d488da3ac49adf25a6b9ac01ae338  \
        --tokenizer-type Llama3Tokenizer \
        --extra-hf-tokens 256 \
        --workers 24 \
        --append-eod

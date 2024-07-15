python3 tools/preprocess_data.py \
        --input /workspace/gpt2/oscar-1GB.jsonl \
        --output-prefix  /workspace/tokenized_data/qwen-2 \
        --hf-tokenizer-path /workspace/pre_trained_hf/tokenizer/base-qwen2-7B  \
        --tokenizer-type Qwen2Tokenizer \
        --extra-hf-tokens 421 \
        --workers 1 \
        --append-eod
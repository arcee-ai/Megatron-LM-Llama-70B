python3 tools/preprocess_data.py \
        --input /workspace/gpt2/oscar-1GB.jsonl \
        --output-prefix  /workspace/Luke-Shamane-processing/tokenized_data/shamane-luke \
        --hf-tokenizer-path /workspace/Luke-Shamane-processing/tokenizer/base-llama3-70B  \
        --tokenizer-type Llama3Tokenizer \
        --extra-hf-tokens 256 \
        --workers 1 \
        --append-eod
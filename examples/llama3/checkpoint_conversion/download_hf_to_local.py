from transformers import AutoModel, AutoTokenizer
import os
import torch

# Specify the Hugging Face model you want to download
# model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
model_name = "meta-llama/Meta-Llama-3-70B"

# Specify your custom directory
custom_dir = "./base-model-hf-checkpoint"

# Create the directory if it doesn't exist
os.makedirs(custom_dir, exist_ok=True)

# Download and save the model directly to the custom directory
model = AutoModel.from_pretrained(model_name, cache_dir=custom_dir, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_dir)

print(f"Model saved in: {custom_dir}")

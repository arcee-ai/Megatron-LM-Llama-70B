docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /fsx/ubuntu/qwen2-SEC-data:/workspace/qwen-2-sec \
  qwe2-preprocessing

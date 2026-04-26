#!/bin/bash

# Inference + assembly pipeline
# By default, outputs both original (ori) and enhanced (xy) images in tiff format with float32 datatype.
# By default, all available GPUs are used for inference. Use CUDA_VISIBLE_DEVICES to specify which GPUs to use.

# === Local ===

# Use all available GPUs
# python test.py --env GHCL00 --scale 4x --override filopodia
python test.py --env GHCL00 --scale 8x --override THX10SDM20xw

# Use specific GPUs
# CUDA_VISIBLE_DEVICES=0,1 python test.py --env GHCL00 --scale 4x --override filopodia
# CUDA_VISIBLE_DEVICES=0,1,2 python test.py --env GHCL00 --scale 8x --override THX10SDM20xw

# Override individual parameters via CLI
# python test.py --env GHCL00 --scale 4x --override THX10SDM20xw model.fp16=True

# Use CPU
# python test.py --env GHCL00 --scale 8x --override THX10SDM20xw --cpu

# === Profiling with NVTX ===
# CUDA_VISIBLE_DEVICES=1 sudo -E env "PATH=$PATH" nsys profile -t cuda,nvtx --gpu-metrics-device=1 -o profile_result speed_test.py --env GHCL00 --scale 8x --override THX10SDM20xw
# sudo -E env "PATH=$PATH" nsys profile -t cuda,nvtx --gpu-metrics-devices=cuda-visible -o profile_result python speed_test.py --env GHCL00 --scale 8x --override THX10SDM20xw
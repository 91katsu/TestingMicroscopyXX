#!/bin/bash

# Inference + assembly pipeline
# By default, outputs both original (ori) and enhanced (xy) images in tiff format with float32 datatype.
# By default, all available GPUs are used for inference. Use CUDA_VISIBLE_DEVICES to specify which GPUs to use.

# === Local ===

# Use all available GPUs
python test_debug.py --env GaryLab10 --config 4xSR --option filopodia

# Use specific GPUs
# CUDA_VISIBLE_DEVICES=0,1 python test.py --env GaryLab10 --config 4xSR --option filopodia

# Use CPU
# python test.py --env GaryLab10 --config 4xSR --option filopodia --cpu

# === Docker ===

# docker run --gpus all \
#   -v /path/to/your/local/data/path:/workspace/data \
#   -v /path/to/your/local/models/path:/workspace/models \
#   -v /path/to/your/local/results/path:/workspace/results \
#   katsukuo/testing-microscopy:v1 \
#   --input_image_filename xxx.tif \
#   --output_dir_name docker_test \
#   --checkpoint_path docker_test
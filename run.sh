#!/bin/bash

# Inference + assembly pipeline
# By default, outputs both original (ori) and enhanced (xy) images in tiff format with float32 datatype.
# By default, all available GPUs are used for inference. Use CUDA_VISIBLE_DEVICES to specify which GPUs to use.

# Output as TIFF (default, uses all available GPUs)
python test.py --gpu --config filopodiaX4 --option ENC

# Use specific GPUs
# CUDA_VISIBLE_DEVICES=0,1 python test.py --gpu --config filopodiaX4 --option ENC

# Output as Zarr
# python test.py --gpu --config filopodiaX4 --option ENC --output_format zarr

# Only want to output enhanced result
# python test.py --gpu --config filopodiaX4 --option ENC --save xy
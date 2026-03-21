#!/bin/bash

# Inference + assembly pipeline
# By default, outputs both original (ori) and enhanced (xy) images in tiff format with float32 datatype.
# By default, all available GPUs are used for inference. Use CUDA_VISIBLE_DEVICES to specify which GPUs to use.

# === Local ===

# Use all available GPUs
python test.py --env GHCL00 --scale 4x --override filopodia
# python test.py --env GHCL00 --scale 8x --override THX10SDM20xw

# Use specific GPUs
# CUDA_VISIBLE_DEVICES=0,1 python test.py --env GHCL00 --scale 8x --override THX10SDM20xw

# Use CPU
# python test.py --env GHCL00 --scale 8x --override THX10SDM20xw --cpu

# Use --override to select a yaml in cfg folder to override DEFAULT settings with a specific section in the YAML
# python test.py --env GHCL00 --scale 4x --override THX10SDM20xw model.epoch=2000 paths.output_dir_name="THX10SDM20xw_epoch2000"
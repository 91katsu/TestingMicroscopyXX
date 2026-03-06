#!/bin/bash

# Inference + assembly pipeline                                     
# By default, outputs both original (ori) and enhanced (xy) images in tiff format with float32 datatype. 

# Output as TIFF (default)
time python test.py --gpu --config filopodiaX4 --option ENC

# Output as Zarr
# time python test.py --gpu --config filopodiaX4 --option ENC --output_format zarr

# Only want to output xy tiff
# time python test.py --gpu --config filopodiaX4 --option ENC --save xy
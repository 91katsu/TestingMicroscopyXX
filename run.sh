# Inference + assembly pipeline                                     
# By default, outputs both original (ori) and enhanced (xy) images in tiff format with float32 datatype. 
# time python test.py --gpu --config filopodiaX4 --option ENC --augmentation encode --output_format zarr
time python test.py --gpu --config filopodiaX4 --option ENC --augmentation encode --save ori xy --output_format tiff --output_datatype float32

# If you want to view Zarr results in a web viewer (e.g. avivator),
# start a local HTTP server from the Zarr root directory.
# For example, if the Zarr path is `result/xxx.zarr`:
#   cd result
#   npx http-server -p 8001 --cors='*'

# After running the above command, you can access the Zarr data in avivator using the URL:
#   http://localhost:8001/xxx.zarr
# Old version (test_only.py)
#python test_only.py --gpu --config DefaultGolgiNovX2 --save ori xy --augmentation encode --testcube --option ENC --savestd -0.7
#python test_assemble.py  --config DefaultGolgiNovX2 --targets ori xy --image_datatype float32 --option ENC

# New simplified version (test_simple.py)
# python test_patch.py --gpu --config DefaultGolgiNovX2 --option ENC --augmentation encode --save ori xy
# python test_assemble.py --config DefaultGolgiNovX2 --targets xy ori --image_datatype float32 --option ENC

# Inference filopodia dataset (output zarr)
# If you want to output tiff files, use "--assemble_method tiff"
time python test_patch.py --gpu --config filopodiaX2 --option ENC --augmentation encode --save xy ori
time python test_assemble.py --config filopodiaX2 --targets xy ori --image_datatype float32 --option ENC --assemble_method zarr

# If you want to view Zarr results in a web viewer (e.g. avivator),
# start a local HTTP server from the Zarr root directory.
# For example, if the Zarr path is `result/xxx.zarr`:
#   cd result
#   npx http-server -p 8001 --cors='*'
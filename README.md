# TestingMicroscopyXX

A deep learning framework for 3D microscopy image super-resolution and reconstruction.

## Overview

This project performs inference and reconstruction on 3D microscopy images using various neural network architectures including Autoencoders, GANs, and Vector Quantized models. It supports patch-based processing for large volumetric data with seamless assembly.

## Features

- **Multiple Model Architectures**: Autoencoders (AE), GANs, VQ-VAE2, CycleGAN/CUT
- **3D Volume Processing**: Patch-based inference with overlapping regions
- **Test-Time Augmentation**: Multiple augmentation strategies for robust predictions
- **Monte Carlo Inference**: Uncertainty estimation via multiple inference runs
- **FP16 Precision**: Efficient mixed-precision inference
- **Seamless Assembly**: Tapered weighting for smooth patch stitching

## Supported Microscopy Modalities

- Structured Illumination Microscopy (SIM)
- Selective Plane Illumination Microscopy (SPIM)
- Golgi apparatus imaging
- Expansion microscopy (iUExM)
- Blood vessel imaging
- Organoid imaging

## Docker Support (Optional)

For a quick start, you can use our pre-built Docker image to run inference on our demo data. See the [Docker Usage Guide](docker_usage.md).

## Project Structure

```
TestingMicroscopyXX/
├── test.py                # Main inference script
├── run.sh                 # Execution examples
├── requirements.txt       # Dependencies
├── cfg/                   # Configuration files
│   ├── env.json           # Environment settings
│   ├── 2xSR.yaml          # 2x super-resolution config
│   ├── 4xSR.yaml          # 4x super-resolution config
│   └── 8xSR.yaml          # 8x super-resolution config
├── models/                # Model definitions
├── networks/              # Neural network architectures
├── utils/                 # Utility modules
└── ldm/                   # Latent Diffusion Model components
```

## Getting Started

### Requirements

- `Python ≥ 3.10`

### Installation
1. Create and Activate a Virtual Environment:
   ```bash
   conda create -n testing-microscopy python=3.10
   conda activate testing-microscopy
   ```
2. Install PyTorch [here](https://pytorch.org/get-started/locally/) according to your system and CUDA configuration.
   ```bash
   # Example: install PyTorch with CUDA 12.6
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run inference and assembly in one step. By default, all available GPUs on the device are used for inference. To specify which GPUs to use, set `CUDA_VISIBLE_DEVICES` (e.g., `CUDA_VISIBLE_DEVICES=0,2` to use GPU 0 and 2):

```bash
# Uses all available GPUs by default
python test.py --env GaryLab10 --config 4xSR --option filopodia

# Use specific GPUs
# CUDA_VISIBLE_DEVICES=0,1 python test.py --env GaryLab00 --config 4xSR --option filopodia

# Run on CPU
# python test.py --env GaryLab00 --config 4xSR --option filopodia --cpu
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Config file name in `cfg/` (without `.yaml`) | required |
| `--option` | Model section in config (overrides `DEFAULT`) | `None` |
| `--env` | Environment name from `env.json` | `None` |
| `--cpu` | Use CPU instead of GPU | off |
| `--input_image_filename` | Input image filename (overrides config) | `None` |
| `--output_dir_name` | Output directory name (overrides config) | `None` |
| `--checkpoint_path` | Checkpoint path (overrides config) | `None` |
| `--epoch` | Model epoch (overrides config) | `None` |

### Configuration

#### Environment (`cfg/env.json`)

Define named environments to set dataset, model, and result paths. Use `--env` to select one at runtime:

```json
{
    "Docker": {
        "DATASET": "/workspace/data/",
        "MODEL": "/workspace/models/",
        "RESULT": "/workspace/results/"
    },
    "GaryLab10": {
        "DATASET": "/path/to/data/",
        "MODEL": "/path/to/models/",
        "RESULT": "/path/to/results/"
    }
}
```

#### YAML Config (`cfg/*.yaml`)

YAML config files have two sections: `DEFAULT` (shared parameters) and an option section selected via `--option`:

```yaml
DEFAULT:
  z_scale_ratio: 4                        # Z-axis scale factor (8 for 8X, 4 for 4X, 2 for 2X)
  testwhole: True                         # process entire volume
  upsample_params:
    size: [64, 256, 256]                  # trilinear upsample target size
  assemble_params:
    C: [16, 16, 16]                       # crop margin per side (pixels)
    S: [16, 16, 16]                       # overlap for tapered blending
    patch_shape: [64, 256, 256]           # inference patch size (Z, Y, X)
    weight_shape: [224, 224, 224]         # patch_shape - C * 2
    zrange: [0, 256, 13 * 4]    # [start, end, step]
    xrange: [0, 1024, 13 * 16]
    yrange: [0, 1024, 13 * 16]
    # If testwhole = True, only the step value matters

  # image setting
  norm_method: ["11"]                     # '00'=as-is, '01'=0-1, '11'=-1~1
  trd: [[ None, None]]                    # intensity clipping thresholds [lower, upper]. [None, None] = no clipping
  norm_percentile: [0.1, 99.9]            # percentile clipping range applied after normalization

  # model setting
  model_type: "VQQ2"                      # AE / GAN / VQQ2
  epoch: 500                              # checkpoint epoch
  hbranchz: true                          # use encoder posterior as h-branch input (VQQ2 only)
  downbranch: 2                           # 1 for 8X SR, 2 for 4X SR, 4 for 2X SR
  checking_codebook: true                 # log VQ codebook usage during inference (VQQ2 only)
  decode_augmentation: false              # apply additional augmentations during decoding
  mc: 1                                   # Monte Carlo passes
  input_augmentation: [null, 'transpose'] # test-time augmentations
  output_channel: 1                       # number of output channels

  # pipeline settings
  fp16: false                             # FP16 mixed-precision inference
  augmentation: "encode"                  # augmentation stage: encode or decode
  save: ["ori", "xy"]                     # targets: ori (upsampled input), xy (enhanced result)
  output_format: "tiff"                   # tiff (per-slice) or zarr (5D volume)
  output_datatype: "float32"              # float32, uint8, or uint16

# use --option filopodia to select this section (overrides DEFAULT)
filopodia:
  input_image_filename: "input.tif"       # DATASET/input_image_filename
  output_dir_name: "filopodia"       # RESULT/output_dir_name
  checkpoint_path: "logs/filopodia/default/max10skip4/"   # MODEL/checkpoint_path

  # any parameter defined here will override the same parameter in DEFAULT
  # For example:
  epoch: 1000
  testwhole: False
```

**Note:** Parameters passed via CLI will override the YAML option section, and the option section will override default section. 

## Workflow

1. **Configure**: Create or select a YAML config file in `cfg/`
2. **Run Inference**: Execute `run.sh` to inference and assembly in one step
3. **Output**: Reconstructed 3D volumes (TIFF or Zarr)
    - Supports uint8, uint16, float32 formats
    - Save targets: `ori` (upsampled input), `xy`(model-enhanced result)
    - Output formats: TIFF (per-slice) or Zarr (5D volume)
    - Default viewing plane is YZ (TIFF saves one YZ slice per X position; Zarr stores X as the primary axis for the same view)
    
## Patch Grid Calculation

The model takes input patches of size `(Z, Y, X) = (256/z_scale_ratio, 256, 256)` and outputs `(256, 256, 256)`. To avoid boundary artifacts, each output patch is cropped by `C` pixels on each side before assembly, resulting in a `(224, 224, 224)` cube. Each cropped patch is then multiplied by a tapered weight that decays toward the edges. Adjacent patches overlap by `S` pixels and are summed in the overlap region for seamless blending. 

The figure below illustrates how the crop size `C` and overlap `S` determine the step size:

![Patch Grid Calculation](assets/TestMicroscopyXX_Patch_Grid_Calculation.png)

**Note:** The step size of `zrange` depends on the Z upscaling factor. If the Z dimension is upscaled by `k×`, the `zrange` step size becomes `1/k` times the `xrange` and `yrange` step size.

## Viewing Zarr Output

To view Zarr results in a web viewer (e.g. [Avivator](https://avivator.gehlenborglab.org/)), start a local HTTP server from the Zarr root directory:

```bash
# If the Zarr path is result/xxx.zarr:
cd result
npx http-server -p 8001 --cors='*'
```

Then open the Zarr data in Avivator using the URL: `http://localhost:8001/xxx.zarr`


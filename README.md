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
├── test.py                  # Main inference script
├── run.sh                   # Execution examples
├── requirements.txt         # Dependencies
├── cfg/                     # Configuration (OmegaConf / YAML)
│   ├── base.yaml            # Default parameters
│   ├── env.yaml             # Machine-specific path settings (DATASET, MODEL, RESULT)
│   ├── scale/               # Scale-specific parameters overrides (patch_shape, zstep, downbranch)
│   │   ├── 2x.yaml          # 2x super-resolution config
│   │   ├── 4x.yaml          # 4x super-resolution config
│   │   └── 8x.yaml          # 8x super-resolution config
│   └── <experiment>.yaml    # Experiment-specific parameters overrides (paths, epochs, etc.)
├── models/                  # Model definitions
├── networks/                # Neural network architectures
├── utils/                   # Utility modules
└── ldm/                     # Latent Diffusion Model components (taming / VQ)
```

### Checkpoint Structure

```
{MODEL}/{ckpt_relpath}/
└── {version}/
    ├── config.json
    ├── {yaml_name}.yaml
    ├── {models}.py
    ├── encoder_model_epoch_{N}.pth
    ├── decoder_model_epoch_{N}.pth
    ├── net_g_model_epoch_{N}.pth
    ├── quantize_model_epoch_{N}.pth
    └── quant_conv_model_epoch_{N}.pth
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

### Configuration

Config is built by layered merging (OmegaConf): `base.yaml` → `scale/{2x,4x,8x}.yaml` → `env.yaml` → `<experiment>.yaml` → `CLI overrides`. Later layers override earlier ones.

#### Environment (`cfg/env.yaml`)

Different machines may store data, models, and results in different locations. Define a named environment for each machine and use `--env` to select one at runtime:

```yaml
Docker:
  DATASET: "/workspace/data/"
  MODEL: "/workspace/models/"
  RESULT: "/workspace/results/"

Machine00:
  DATASET: "/path/to/Machine00/Data/"
  MODEL: "/path/to/Machine00/Model/"
  RESULT: "/path/to/Machine00/Result/"

Machine01:
  DATASET: "/path/to/Machine01/Data/"
  MODEL: "/path/to/Machine01/Model/"
  RESULT: "/path/to/Machine01/Result/"
```

#### Base config (`cfg/base.yaml`)

Defines all available parameters and their default values. Later config layers can override any of these.

**`patch`** — Patch dimensions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_shape` | `null` (Set by scale config) | Inference patch size `[Z, Y, X]` |
| `upsample_size` | Same as `patch_shape` | Upsample target size for input patch before inference |

**`assemble`** — Assembly parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `[16, 16, 16]` | Patch crop size (pixels) per side `[Z, Y, X]` |
| `S` | `[16, 16, 16]` | Patch overlap size (pixels) for tapered blending `[Z, Y, X]` |
| `weight_shape` | `[224, 224, 224]` | Tapered weight shape for blending (`patch_shape - 2*C`) |

**`grid`** — Patch grid and ROI

| Parameter | Default | Description |
|-----------|---------|-------------|
| `testwhole` | `True` | `True`: process entire volume; `False`: crop to ROI first |
| `roi.z/y/x` | `[0, 256]` / `[0, 1024]` / `[0, 1024]` | ROI range (only used when `testwhole: False`) |
| `step.z` | `null` (Set by scale config) | Z-axis step size |
| `step.y` | `13 * 16` | Y-axis step size |
| `step.x` | `13 * 16` | X-axis step size |

**`preprocess`** — Data preprocessing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `norm_method` | `["00"]` | Normalization: `"00"` = as-is, `"01"` = 0–1, `"11"` = -1–1 |
| `trd` | `[[0, 1000]]` | Intensity clipping `[lower, upper]` |
| `norm_percentile` | `[0.1, 99.9]` | Percentile clipping range |

**`model`** — Model settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | `"VQQ2"` | Model type: `AE`, `GAN`, `VQQ2` |
| `epoch` | `500` | Checkpoint epoch to load |
| `hbranchz` | `true` | Use encoder posterior as h-branch input (VQQ2 only) |
| `downbranch` | `null` (Set by scale config) | Downsampling factor. `1` for 8X SR, `2` for 4X SR, `4` for 2X SR |
| `checking_codebook` | `true` | Log VQ codebook usage during inference (VQQ2 only) |
| `decode_augmentation` | `false` | Apply augmentations during decoding |
| `num_mc` | `1` | Number of Monte Carlo passes (repeats `tta_method` list `num_mc` times) |
| `mc_threshold` | `null` | Threshold for binary MC std map. Required when `num_mc > 1` |
| `tta_mode` | `"encode"` | Test-time augmentation stage: `encode` or `decode` |
| `tta_method` | `[null, "transpose"]` | Test-time augmentation methods |
| `fp16` | `false` | FP16 mixed-precision inference |

**`output`** — Output settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save` | `["ori", "xy"]` | Targets to save: `ori` (upsampled input), `xy` (enhanced result), `xystd` (MC uncertainty map, uint8) |
| `output_format` | `"tiff"` | `tiff` (per-slice) or `zarr` (5D volume) |
| `output_datatype` | `"float32"` | `float32`, `uint16`, `uint8` |
| `output_channel` | `1` | Number of output channels |

**`paths`** — Input/output paths (required, must be set by experiment config or CLI)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_img_relpath` | `null` | Input image path relative to `DATASET` |
| `ckpt_relpath` | `null` | Checkpoint directory relative to `MODEL` |
| `version` | `null` | Version subdirectory under `ckpt_relpath` |
| `output_dir_name` | `null` | Output directory name under `RESULT` |

Runtime paths (`input_img_path`, `ckpt_root_path`, `output_dir`) are built automatically via `os.path.join(env.*, paths.*)` in the code.

#### Scale config (`cfg/scale/{2x,4x,8x}.yaml`)

Sets scale-dependent parameters (`patch_shape`, `grid.step.z`, `model.downbranch`). 
Example (`cfg/scale/2x.yaml`):

```yaml
patch:
  patch_shape: [128, 256, 256]
grid:
  step:
    z: 13 * 16 / 2
model:
  downbranch: 4
```

#### Experiment config (`cfg/<experiment>.yaml`)

Override any parameter for a specific experiment. Use `--override <name>` to apply:

```yaml
grid:
  testwhole: False
  roi:
    z: [0, 256]
    y: [0, 2048]
    x: [0, 2048]
model:
  epoch: 1700
output:
  output_datatype: "uint8"
paths:
  input_img_relpath: "THX10SDM20xw/roiDcrop2.tif"
  ckpt_relpath: "THX10SDM20xw/max5skip4/checkpoints"
  output_dir_name: "THX10SDM20xw"
```

#### CLI overrides

Any parameter can be overridden directly from the command line using dot notation. CLI overrides are applied last, taking the highest priority:

```bash
python test.py --env GHCL00 --scale 4x --override THX10SDM20xw model.epoch=2000 paths.output_dir_name="THX10SDM20xw_epoch2000"
```

### Usage

Run inference and assembly in one step. By default, all available GPUs on the device are used for inference. To specify which GPUs to use, set `CUDA_VISIBLE_DEVICES` (e.g., `CUDA_VISIBLE_DEVICES=0,2` to use GPU 0 and 2):

```bash
# Use all available GPUs
python test.py --env GHCL00 --scale 8x --override THX10SDM20xw

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python test.py --env GHCL00 --scale 8x --override THX10SDM20xw

# Use CPU
python test.py --env GHCL00 --scale 8x --override THX10SDM20xw --cpu

# Override individual parameters via CLI
python test.py --env GHCL00 --scale 4x --override THX10SDM20xw model.epoch=2000 paths.output_dir_name="THX10SDM20xw_epoch2000"
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--env` | Environment name from `env.yaml` (e.g. `Docker`, `GHCL00`) | required |
| `--scale` | Scale config from `cfg/scale/` (`2x`, `4x`, `8x`) | required |
| `--override` | Experiment config in `cfg/` (without `.yaml`) | `None` |
| `--cpu` | Use CPU instead of GPU | off |
| `key=value` | Additional CLI overrides (dot notation, e.g. `model.epoch=2000`) | — |

### Results

- Results will be saved to: `{RESULT}/{paths.output_dir_name}/`
- The output directory contains the targets specified in `output.save`:
  - `ori`: upsampled original image
  - `xy`: model-enhanced image
  - `xystd`: Monte Carlo uncertainty map (uint8)
  - `config.yaml`: resolved config snapshot (always saved)
- Output is saved as YZ-plane slices along the X axis
- Format (`output.output_format`): `tiff`, `zarr`
- Datatype (`output.output_datatype`): `float32`, `uint16`, `uint8`
- Viewing Zarr output: 
  1. Start a local HTTP server from the Zarr root directory:
      ```bash
      # If the Zarr path is result/xxx.zarr
      cd result
      npx http-server -p 8001 --cors='*'
      ```
  2. Then open `http://localhost:8001/xxx.zarr` in [Avivator](https://avivator.gehlenborglab.org).

## Monte Carlo Uncertainty Estimation

If you need Monte Carlo uncertainty boundary maps, see [here](monte_carlo.md) for usage instructions.
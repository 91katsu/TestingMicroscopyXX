# Docker Usage

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (for GPU support)

## Quick Start
1. Download the demo testing package [here](https://drive.google.com/drive/folders/1eOBwrdz3DiwsjrEF2ZQNeeaNbYcsa7_K?usp=sharing), then extract it to your cloud storage

    ```
    /docker_test_demo/
    ├──input/
    │   ├── max5skip4/                          # model folder
    │   │   ├── 0.json                          
    │   │   ├── ae0iso0tccutvqq.py              
    │   │   ├── checkpoints/
    │   │   │   ├── encoder_model_epoch_1700.pth
    │   │   │   ├── decoder_model_epoch_1700.pth
    │   │   │   ├── net_g_model_epoch_1700.pth
    │   │   │   ├── netF_model_epoch_1700.pth
    │   │   │   ├── quant_conv_model_epoch_1700.pth
    │   │   │   ├── post_quant_conv_model_epoch_1700.pth
    │   │   │   └── quantize_model_epoch_1700.pth
    │   │   └── logs/
    │   ├── roiDcrop2.tif                       # input image
    │   └── thxvqqbrc.yaml                      # config
    │
    └──output/
    ```

2. Pull the Docker Image

    ```bash
    docker pull katsukuo/testing-microscopy:v3
    ```

3. Run the container (replace `<cloud_storage_path>` with your actual storage path)
    ```bash
    docker run --gpus all \
      -v /<cloud_storage_path>/docker_test_demo/input:/workspace/data \
      -v /<cloud_storage_path>/docker_test_demo/input:/workspace/models \
      -v /<cloud_storage_path>/docker_test_demo/input/thxvqqbrc.yaml:/workspace/cfg/thxvqqbrc.yaml \
      -v /<cloud_storage_path>/docker_test_demo/output:/workspace/results \
      katsukuo/testing-microscopy:v3 \
      --config thxvqqbrc \
      --option THX10SDM20xw
    ```

    Note that the container expects four volume mounts:

    | Cloud Server Path | Container Path | Description |
    |---|---|---|
    | `/<cloud_storage_path>/docker_test_demo/input/` | `/workspace/data/` | Input image directory (TIFF files) |
    | `/<cloud_storage_path>/docker_test_demo/input/` | `/workspace/models/` | Model checkpoints root directory |
    | `/<cloud_storage_path>/docker_test_demo/output/` | `/workspace/results/` | Output root directory |
    | `/<cloud_storage_path>/docker_test_demo/xxx.yaml/` | `/workspace/cfg/xxx.yaml` | config yaml file |

    - `--config` specifies which YAML config file to use (without the `.yaml` extension).
    - `--option` selects the dataset/option section defined in the config file.

4. Outputs

    Results will be saved to `/<cloud_storage_path>/docker_test_demo/output/THX10SDM20xw/` with the following default behavior:

    - **Format:** TIFF (`uint8`)
    - **Output:** 
      - Both original (`ori`) and enhanced (`xy`) images
      - inference config file
      ```
      /cloud_storage/
      ├──input/
      └──output/
          ├── config.yaml
          ├── ori_assemble_0/
          └── xy_assemble_0/
      ```
import argparse
import json
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff
import yaml
import zarr
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser
from utils.data_utils import DataNormalization

import taming.modules.vqvae.quantize as q
if not hasattr(q, "VectorQuantizer2"):
    if hasattr(q, "VectorQuantizer"):
        q.VectorQuantizer2 = q.VectorQuantizer

# =============================================================================
# Config
# =============================================================================
class Config:
    """Wraps YAML config loading and access."""

    def __init__(self, config_name, option=None, env=None, overrides=None):
        self.config_name = config_name
        self.option = option
        self.env = env
        self.overrides = overrides or {}
        self.cfg = self._load()

    def _load(self):
        with open(f'cfg/{self.config_name}.yaml', 'r') as f:
            config = yaml.safe_load(f)

        if self.option is not None:
            cfg = {**config['DEFAULT'], **config[self.option]}
        else:
            cfg = {**config['DEFAULT']}

        if self.env is not None:
            with open('cfg/env.json', 'r') as f:
                env_cfg = json.load(f)[self.env]
            cfg['MODEL'] = env_cfg['MODEL']
            cfg['root_path'] = env_cfg['DATASET']
            cfg['DESTINATION'] = env_cfg['RESULT']

        # CLI overrides
        for key, value in self.overrides.items():
            if value is not None:
                cfg[key] = value

        # Validate required fields
        required = ['input_image_filename', 'output_dir_name', 'checkpoint_path']
        missing = [k for k in required if k not in cfg or cfg[k] is None]
        if missing:
            raise ValueError(f"Missing required fields: {missing}. Set them in YAML (--option) or via CLI arguments.")

        return cfg

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def __getitem__(self, key):
        return self.cfg[key]


# =============================================================================
# ModelLoader
# =============================================================================
class VQQModel:
    """Container for VQQ2 model components."""

    def __init__(self, ckpt, epoch):
        self.encoder = torch.load(os.path.join(ckpt, f"encoder_model_epoch_{epoch}.pth"), map_location='cpu', weights_only=False)
        self.decoder = torch.load(os.path.join(ckpt, f"decoder_model_epoch_{epoch}.pth"), map_location='cpu', weights_only=False)
        self.net_g = torch.load(os.path.join(ckpt, f"net_g_model_epoch_{epoch}.pth"), map_location='cpu', weights_only=False)
        self.quantize = torch.load(os.path.join(ckpt, f"quantize_model_epoch_{epoch}.pth"), map_location='cpu', weights_only=False)
        self.quant_conv = torch.load(os.path.join(ckpt, f"quant_conv_model_epoch_{epoch}.pth"), map_location='cpu', weights_only=False)
        self.post_quant_conv = torch.load(os.path.join(ckpt, f"post_quant_conv_model_epoch_{epoch}.pth"), map_location='cpu', weights_only=False)

    def to(self, device):
        for attr in ['encoder', 'decoder', 'net_g', 'quantize', 'quant_conv', 'post_quant_conv']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def cuda(self):
        return self.to('cuda')

    def half(self):
        for attr in ['encoder', 'decoder', 'net_g', 'quantize', 'quant_conv', 'post_quant_conv']:
            setattr(self, attr, getattr(self, attr).half())
        return self

    def parameters(self):
        for attr in ['encoder', 'decoder', 'net_g', 'quantize', 'quant_conv', 'post_quant_conv']:
            for p in getattr(self, attr).parameters():
                yield p


class ModelLoader:
    """Handles model loading for different model types."""

    def __init__(self, cfg, device='cpu', fp16=False, augmentation='encode'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.gpu = (self.device.type == 'cuda')
        self.fp16 = fp16
        self.augmentation = augmentation

        self.model = self._load_model()
        self.upsample = torch.nn.Upsample(size=cfg['upsample_params']['size'], mode='trilinear')

        if self.gpu:
            self.upsample = self.upsample.to(self.device)

        self.model_proc = ModelProcesser(
            cfg.cfg if isinstance(cfg, Config) else cfg,
            self.model,
            gpu=self.gpu,
            augmentation=augmentation,
            fp16=fp16,
            device=self.device
        )

    def _load_model(self):
        cfg = self.cfg.cfg if isinstance(self.cfg, Config) else self.cfg
        ckpt_root = os.path.join(cfg['MODEL'], cfg['checkpoint_path'])
        epoch = cfg['epoch']
        model_type = cfg['model_type']
        print(f"Loading model from checkpoint, epoch: {epoch}, model type: {model_type}")

        if model_type == 'AE':
            model = self._load_ae(ckpt_root, epoch)
        elif model_type == 'GAN':
            model = self._load_gan(ckpt_root, epoch)
        elif model_type == 'VQQ2':
            model = self._load_vqq2(ckpt_root, epoch)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if self.gpu:
            model = model.to(self.device) if hasattr(model, 'to') else model
        for p in model.parameters():
            p.requires_grad = False
        if self.fp16:
            model = model.half()

        return model

    def _load_ae(self, ckpt_root, epoch):
        args = read_json_to_args(os.path.join(ckpt_root, "0.json"))
        model_module = import_model(ckpt_root, model_name=args.models)
        model = model_module.GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
        model = load_pth(model, root=ckpt_root, epoch=epoch,
                         model_names=['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv'])
        return model

    def _load_gan(self, ckpt_root, epoch):
        model_path = os.path.join(ckpt_root, f"checkpoints/net_g_model_epoch_{epoch}.pth")
        return torch.load(model_path, map_location='cpu')

    def _load_vqq2(self, ckpt_root, epoch):
        ckpt = os.path.join(ckpt_root, "checkpoints")
        return VQQModel(ckpt, epoch)


# =============================================================================
# ImageLoader
# =============================================================================
class ImageLoader:
    """Handles image loading and preprocessing."""

    def __init__(self, cfg):
        self.cfg = cfg.cfg if isinstance(cfg, Config) else cfg
        self.normalizer = DataNormalization(backward_type='float32')
        self.img = None

    def load(self):
        image_path = self.cfg['root_path'] + self.cfg['input_image_filename']
        print(f"Loading: {image_path}")
        img = tiff.imread(image_path) # (Z, Y, X) # (237, 2763, 2756)
        self.img = self.normalizer.forward_normalization(
            img, self.cfg['norm_method'][0], self.cfg['trd'][0]
        ) # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)
        return self.img

    def apply_percentile_norm(self):
        if self.img is None:
            raise ValueError("Image not loaded. Call load() first.")

        if self.cfg.get('norm_percentile') is not False:
            p0, p1 = self.cfg.get('norm_percentile', [0.5, 99.5])
            if not isinstance(p0, (int, float)):
                p0, p1 = 0.5, 99.5

            xmin = np.percentile(self.img[..., ::2, ::2].flatten(), p0)
            xmax = np.percentile(self.img[..., ::2, ::2].flatten(), p1)
            print(f"Percentile clipping: [{xmin:.2f}, {xmax:.2f}]")

            self.img = torch.clamp(self.img, xmin, xmax)
            self.img = (self.img - xmin) / (xmax - xmin)
            self.img = self.img * 2 - 1

        return self.img # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)

    def pad(self, crop_margin=None):
        """
        Pad image for patch-based inference.

        Args:
            crop_margin: (Cz, Cy, Cx) pixels cropped from each side during assembly.
                         If provided, pads by C on both sides to preserve original size.
        """
        if self.img is None:
            raise ValueError("Image not loaded. Call load() first.")

        _, _, Dz, Dy, Dx = self.img.shape # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)

        print(f"Padding image from {self.img.shape}")

        # First, pad by crop margin (C) on both sides to preserve original size after assembly
        if crop_margin is not None:
            Cz, Cy, Cx = crop_margin
            self.img = F.pad(self.img, (Cx, Cx, Cy, Cy, Cz//self.cfg['z_scale_ratio'] , Cz//self.cfg['z_scale_ratio']), mode='constant', value=self.img.min())
            _, _, Dz, Dy, Dx = self.img.shape # (1, 1, Z, Y, X) # (1, 1, 245, 2795, 2788)
            print(f"After C padding: {self.img.shape}")

        zstep = int(eval(self.cfg['assemble_params']['zrange'][-1]))
        ystep = int(eval(self.cfg['assemble_params']['yrange'][-1]))
        xstep = int(eval(self.cfg['assemble_params']['xrange'][-1]))
        # Then, pad to make divisible by patch shape
        Nz = ((Dz // zstep) + 1) * zstep
        Ny = ((Dy // ystep) + 1) * ystep
        Nx = ((Dx // xstep) + 1) * xstep

        Pz, Py, Px = Nz - Dz, Ny - Dy, Nx - Dx
        self.img = F.pad(self.img, (0, Px, 0, Py, 0, Pz), mode='constant', value=self.img.min())
        print(f"After stride padding: {self.img.shape}")

        return self.img # (1, 1, Z, Y, X) # (1, 1, 260, 2912, 2912)


# =============================================================================
# PatchProcessor
# =============================================================================
class PatchProcessor:
    """Handles inference on patches."""

    def __init__(self, model_proc, upsample, augmentations, fp16=False, gpu=False, device='cpu'):
        self.model_proc = model_proc
        self.upsample = upsample
        self.augmentations = augmentations
        self.fp16 = fp16
        self.gpu = gpu
        self.device = torch.device(device)

    def run(self, x0_list, startIdx_zyx, patchShape_zyx, ii=None):
        # Extract and upsample patch
        # x0_list is on CPU, slice then move to this worker's device
        # (B, C, Z, Y, X) # (1, 1, 64, 256, 256)
        patch = [x[:, :, startIdx_zyx[0]:startIdx_zyx[0]+patchShape_zyx[0], startIdx_zyx[1]:startIdx_zyx[1]+patchShape_zyx[1], startIdx_zyx[2]:startIdx_zyx[2]+patchShape_zyx[2]] for x in x0_list]
        # Move to this worker's device for upsample
        patch = [x.to(self.device) for x in patch]
        # (Z, C, Y, X) # (64, 1, 256, 256)
        patch = torch.cat([self.upsample(x).squeeze().unsqueeze(1) for x in patch], 1)

        if self.fp16 and self.gpu:
            patch = patch.half()
            with torch.cuda.amp.autocast():
                _, Xup, outall, _ = self.model_proc.get_model_result(patch, self.augmentations, ii=ii)
        else:
            # Xup: (Z, C, Y, X) # (256, 1, 256, 256)
            # outall: (Z, C, Y, X, aug) # (256, 1, 256, 256, 2)
            _, Xup, outall, _ = self.model_proc.get_model_result(patch, self.augmentations, ii=ii)

        outall = outall.numpy().astype(np.float32) # (Z, C, Y, X, aug) # (256, 1, 256, 256, 2)
        Xup = Xup.numpy().astype(np.float32) # (Z, C, Y, X) # (256, 1, 256, 256)

        # Match output range to input range per channel
        for c in range(outall.shape[1]):
            omin, omax = outall[:, c, :, :, :].min(), outall[:, c, :, :, :].max()
            xmin, xmax = Xup[:, c, :, :].min(), Xup[:, c, :, :].max()
            outall[:, c, :, :, :] = (outall[:, c, :, :, :] - omin) / (omax - omin + 1e-6) * (xmax - xmin) + xmin
        # Xup: (Z, C, Y, X) # (256, 1, 256, 256)
        # outall: (Z, C, Y, X, aug) # (256, 1, 256, 256, 2)
        return outall, Xup


# =============================================================================
# PatchAssembler
# =============================================================================
class PatchAssembler:

    def __init__(self, cfg, dest, target_name, output_format, output_datatype,
                 zrange, xrange, yrange, output_channel):
        self.Cz, self.Cy, self.Cx = cfg['assemble_params']['C']
        self.Sz, self.Sy, self.Sx = cfg['assemble_params']['S']
        self.weight_shape = cfg['assemble_params']['weight_shape']
        self.output_datatype = output_datatype
        self.output_format = output_format
        self.output_channel = output_channel
        self.zrange = zrange
        self.xrange = xrange
        self.yrange = yrange

        self.patches = {}
        self.last_block = None
        self.current_x_position = 0

        # Setup output
        self.output_path = os.path.join(dest, target_name + '_assemble')
        if output_format == 'tiff':
            self._init_tiff_dirs()
        elif output_format == 'zarr':
            self._init_zarr_store()

    def _init_tiff_dirs(self):
        for c in range(self.output_channel):
            d = self.output_path + '_' + str(c)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    def _init_zarr_store(self):
        wZ, wX, wY = self.weight_shape
        nz = len(self.zrange)
        ny = len(self.yrange)
        nx = len(self.xrange)
        fZ = int(wZ * nz - (nz - 1) * self.Sz)
        fY = int(wY * ny - (ny - 1) * self.Sy)
        fX = int(wX * nx - (nx - 1) * self.Sx - self.Sx)

        C = self.output_channel
        save_dtype = np.dtype(self.output_datatype)

        # TCZYX 5D for OME-Zarr: store X as zarr Z-axis (side view)
        shape_5d = (1, C, fX, fZ, fY)
        chunks_5d = (1, 1, 20, 512, 512)

        zarr_path = self.output_path + '.zarr'
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.open_group(store, mode="w")
        self.zarr_out = root.create_dataset(
            "0", shape=shape_5d, chunks=chunks_5d, dtype=save_dtype, fill_value=0
        )
        self.zarr_path = zarr_path
        print(f"[zarr] Created store: {zarr_path}, shape TCZYX={shape_5d}")

    def store_patch(self, nz, ny, data, is_outall=False):
        """Store a patch for the current X loop.

        Args:
            nz, ny: grid indices within current X loop
            data: outall (Z,C,Y,X,aug) if is_outall, else Xup (Z,C,Y,X)
            is_outall: True if data has augmentation dimension to average
        """
        if is_outall:
            data = data.mean(axis=-1)  # average augmentations => (Z, C, Y, X, aug) -> (Z, C, Y, X)
        patch = np.transpose(data, (1, 0, 2, 3)).astype(np.float32)  # (C, Z, Y, X)
        self.patches[(nz, ny)] = patch

    def _convert_dtype(self, data):
        """Convert float32 data (roughly -1 to 1 range) to target dtype."""
        if self.output_datatype == 'uint8':
            return np.clip((data + 1) * 127.5, 0, 255).astype(np.uint8)
        elif self.output_datatype == 'uint16':
            return np.clip((data + 1) * 32767.5, 0, 65535).astype(np.uint16)
        return data.astype(np.float32)

    def create_tapered_weight(self, Sz, Sy, Sx, nz, ny, nx, size, edge_size: int = 64) -> np.ndarray:
        """Create a 3D cube with linearly tapered edges in all directions.

        Weight axes: 0 → Z (tapered by Sz), 1 → (tapered by Sy), 2 → (tapered by Sx).
        Edge flags: 0 = first patch (no taper on leading edge),
                -1 = last patch (no taper on trailing edge).
        """
        weight = np.ones(size)

        taper_Sz = np.linspace(0, 1, Sz)
        taper_Sy = np.linspace(0, 1, Sy)
        taper_Sx = np.linspace(0, 1, Sx)
        # Z
        if nz != 0:
            weight[:Sz, :, :] *= taper_Sz.reshape(-1, 1, 1)
        if nz != -1:
            weight[-Sz:, :, :] *= taper_Sz[::-1].reshape(-1, 1, 1)
        # Y
        if ny != 0:
            weight[:, :Sy, :] *= taper_Sy.reshape(1, -1, 1)
        if ny != -1:
            weight[:, -Sy:, :] *= taper_Sy[::-1].reshape(1, -1, 1)
        # X
        if nx != 0:
            weight[:, :, :Sx] *= taper_Sx.reshape(1, 1, -1)
        if nx != -1:
            weight[:, :, -Sx:] *= taper_Sx[::-1].reshape(1, 1, -1)

        return weight

    def assemble_column(self, nx, patches=None):
        """Assemble all patches for X-column index nx and write output.

        Args:
            nx: X-column index.
            patches: dict of {(nz, ny): patch_array}. If None, uses self.patches.

        Uses zarr version's axis convention:
        - Inner loop (ny): blend along Y (axis 3 of CZXY)
        - Middle loop (nz): blend along Z (axis 2 after transpose to CXZY)
        - Outer (nx): blend along X (axis 1) via last_block
        """
        if patches is None:
            patches = self.patches
        Cz, Cy, Cx = self.Cz, self.Cy, self.Cx
        Sz, Sy, Sx = self.Sz, self.Sy, self.Sx
        nz_count = len(self.zrange)
        ny_count = len(self.yrange)
        nx_count = len(self.xrange)

        one_zy_block = []
        for nz in range(nz_count):
            one_column = []
            for ny in range(ny_count):
                # Edge flags: 0 = first patch, -1 = last patch
                nz_flag = -1 if nz == nz_count - 1 else nz
                nx_flag = -1 if nx == nx_count - 1 else nx
                ny_flag = -1 if ny == ny_count - 1 else ny

                # Get patch and crop margins from all sides
                patch = patches[(nz, ny)]  # (C, Z, X, Y) # (1, 256, 256, 256)
                cropped = patch[:, Cz:-Cz, Cy:-Cy, Cx:-Cx] # (C, Z, X, Y) # (1, 224, 224, 224)

                w = self.create_tapered_weight(Sz, Sy, Sx, nz_flag, ny_flag, nx_flag,
                                          size=self.weight_shape)
                w = np.stack([w] * cropped.shape[0], axis=0)
                cropped = np.multiply(cropped, w) # (C, Z, X, Y) # (1, 224, 224, 224)

                # Y-axis blending: overlap Sy along axis 2
                if len(one_column) > 0:
                    one_column[-1][:, :, -Sy:, :] += cropped[:, :, :Sy, :]
                    one_column.append(cropped[:, :, Sy:, :])
                else:
                    one_column.append(cropped)

            # Concat one column along Y (axis 2), then reorder to (C, X, Z, Y)
            one_column = np.concatenate(one_column, axis=2)  # (C, Z, Y_total, X)
            one_column = np.transpose(one_column, (0, 3, 1, 2))  # (C, X, Z, Y_total)

            # Z-axis blending: overlap Sz along axis 2
            if len(one_zy_block) > 0:
                one_zy_block[-1][:, :, -Sz:, :] += one_column[:, :, :Sz, :]
                one_zy_block.append(one_column[:, :, Sz:, :])
            else:
                one_zy_block.append(one_column)

        one_zy_block = np.concatenate(one_zy_block, axis=2).astype(np.float32)  # (C, X, Z_total, Y_total)

        # X-axis blending: overlap Sx along axis 1
        if self.last_block is not None:
            one_zy_block[:, :Sx, :, :] += self.last_block

        # Write output (excluding the trailing Sx overlap)
        if self.output_format == 'tiff':
            self._write_column_tiff(one_zy_block, Sx)
        elif self.output_format == 'zarr':
            self._write_column_zarr(one_zy_block, Sx)

        # Save overlap for next column
        self.last_block = one_zy_block[:, -Sx:, :, :].copy()
        self.current_x_position += one_zy_block.shape[1] - Sx

    def _write_column_tiff(self, one_zy_block, Sx):
        """Write assembled column as per-X-position TIFF slices."""
        for x in range(0, one_zy_block.shape[1] - Sx):
            for c in range(one_zy_block.shape[0]):
                slice_data = one_zy_block[c, x, :, :]
                slice_data = self._convert_dtype(slice_data)
                tiff.imwrite(
                    os.path.join(self.output_path + '_' + str(c),
                                 'slice_x_' + str(self.current_x_position + x).zfill(4) + '.tif'),
                    slice_data
                )

    def _write_column_zarr(self, one_zy_block, Sx):
        """Write assembled column block to zarr store."""
        block = one_zy_block[:, :one_zy_block.shape[1] - Sx, :, :]
        block = self._convert_dtype(block)
        x_len = block.shape[1]
        self.zarr_out[0, :, self.current_x_position:self.current_x_position + x_len, :, :] = block


# =============================================================================
# InferencePipeline
# =============================================================================
class InferencePipeline:
    """patch inference + assembly pipeline."""

    def __init__(self, args):
        self.args = args
        overrides = {
            'input_image_filename': args.input_image_filename,
            'output_dir_name': args.output_dir_name,
            'checkpoint_path': args.checkpoint_path,
            'epoch': args.epoch,
        }
        self.config = Config(args.config, args.option, env=args.env, overrides=overrides)

        # Read pipeline settings from config
        self.fp16 = self.config.get('fp16', False)
        self.augmentation = self.config.get('augmentation', 'encode')
        self.save_targets = self.config.get('save', ['ori', 'xy'])
        self.output_format = self.config.get('output_format', 'tiff')
        self.output_datatype = self.config.get('output_datatype', 'float32')

        # Setup output directory
        self.dest = os.path.join(self.config['DESTINATION'], self.config['output_dir_name'])
        self._setup_output_dirs()

        # Determine devices
        if args.cpu:
            self.num_gpus = 1
            self.devices = ['cpu']
        else:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus == 0:
                raise RuntimeError("No CUDA devices found. Use --cpu to run on CPU.")
            self.devices = [f'cuda:{i}' for i in range(self.num_gpus)]

        print(f"Using {self.num_gpus} device(s): {self.devices}")

        # Build one model + processor per device
        self.workers = []
        for device in self.devices:
            print(f"Loading model on {device}...")
            loader = ModelLoader(
                self.config,
                device=device,
                fp16=self.fp16,
                augmentation=self.augmentation
            )
            proc = PatchProcessor(
                loader.model_proc,
                loader.upsample,
                self.config.get('input_augmentation', [None]),
                fp16=self.fp16,
                gpu=(device != 'cpu'),
                device=device
            )
            self.workers.append(proc)

        print("Loading image...")
        self.image_loader = ImageLoader(self.config)

    def _setup_output_dirs(self):
        os.makedirs(self.dest, exist_ok=True)
        # Save config
        with open(os.path.join(self.dest, 'config.yaml'), 'w') as f:
            yaml.dump(self.config.cfg, f)

    def _get_patch_grid(self, img_shape):
        cfg = self.config.cfg
        _, _, zz, yy, xx = img_shape

        if cfg.get('testwhole', True):
            zstep = int(eval(cfg['assemble_params']['zrange'][-1]))
            ystep = int(eval(cfg['assemble_params']['yrange'][-1]))
            xstep = int(eval(cfg['assemble_params']['xrange'][-1]))
            zrange = range(0, zz, zstep)
            yrange = range(0, yy, ystep)
            xrange = range(0, xx, xstep)
        else:
            zrange = range(*[int(eval(str(x))) for x in cfg['assemble_params']['zrange']])
            yrange = range(*[int(eval(str(x))) for x in cfg['assemble_params']['yrange']])
            xrange = range(*[int(eval(str(x))) for x in cfg['assemble_params']['xrange']])

        return zrange, yrange, xrange

    def run(self):
        # Load and preprocess image
        self.image_loader.load() # (B, C, Z, Y, X) # (1, 1, 237, 2763, 2756)
        self.image_loader.apply_percentile_norm() # (B, C, Z, Y, X) # (1, 1, 237, 2763, 2756)
        patch_shape = self.config['assemble_params']['patch_shape']
        dz, dy, dx = patch_shape

        # Get crop margin C for padding (to preserve original size after assembly)
        crop_margin = None
        if self.config.cfg.get('testwhole', True):
            crop_margin = self.config['assemble_params']['C']

        self.image_loader.pad(crop_margin=crop_margin) # (B, C, Z, Y, X) # (1, 1, 260, 2912, 2912)
        x0 = [self.image_loader.img]

        # Get patch grid
        zrange, yrange, xrange = self._get_patch_grid(self.image_loader.img.shape)
        total_patches = len(zrange) * len(yrange) * len(xrange)
        print(f"Processing {len(zrange)}x{len(yrange)}x{len(xrange)} = {total_patches} patches")

        # Create one assembler per save target
        assemblers = {}
        for target in self.save_targets:
            assemblers[target] = PatchAssembler(
                cfg=self.config.cfg,
                dest=self.dest,
                target_name=target,
                output_format=self.output_format,
                output_datatype=self.output_datatype,
                zrange=zrange, xrange=xrange, yrange=yrange,
                output_channel=self.config.get('output_channel', 1)
            )

        print('xrange:', xrange)
        print('zrange:', zrange)
        print('yrange:', yrange)

        # Build list of (nz, ny, iz, iy) for each x-column
        column_patches = []
        for nz, iz in enumerate(zrange):
            for ny, iy in enumerate(yrange):
                column_patches.append((nz, ny, iz, iy))

        num_workers = len(self.workers)
        # Use a persistent thread pool + a separate thread for assembly
        assemble_executor = ThreadPoolExecutor(max_workers=1)
        assemble_future = None

        with tqdm(total=total_patches, desc="Processing") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as infer_executor:
                for nx, ix in enumerate(xrange):
                    futures = {}
                    for idx, (nz, ny, iz, iy) in enumerate(column_patches):
                        gpu_id = idx % num_workers
                        fut = infer_executor.submit(
                            self.workers[gpu_id].run,
                            x0,
                            startIdx_zyx=[iz, iy, ix],
                            patchShape_zyx=[dz, dy, dx],
                            ii=(iz, iy, ix)
                        )
                        futures[fut] = (nz, ny)

                    # Collect results (order doesn't matter for store_patch)
                    for fut in as_completed(futures):
                        nz, ny = futures[fut]
                        outall, Xup = fut.result()
                        # outall: (Z, C, Y, X, aug) # (256, 1, 256, 256, 2)
                        # Xup: (Z, C, Y, X) # (256, 1, 256, 256)
                        if 'xy' in assemblers:
                            assemblers['xy'].store_patch(nz, ny, outall, is_outall=True)
                        if 'ori' in assemblers:
                            assemblers['ori'].store_patch(nz, ny, Xup, is_outall=False)
                        pbar.update(1)

                    # Snapshot patches and reset — so next column can store immediately
                    patches_snapshots = {}
                    for target, asm in assemblers.items():
                        patches_snapshots[target] = asm.patches
                        asm.patches = {}  # fresh dict for next column

                    # Wait for previous assembly (it uses last_block for X-blending)
                    if assemble_future is not None:
                        assemble_future.result()

                    # Launch assembly in background with snapshot
                    def _assemble(assemblers, snapshots, nx):
                        for target, asm in assemblers.items():
                            asm.assemble_column(nx, patches=snapshots[target])
                    assemble_future = assemble_executor.submit(
                        _assemble, assemblers, patches_snapshots, nx
                    )

        # Wait for the last column's assembly
        if assemble_future is not None:
            assemble_future.result()
        assemble_executor.shutdown(wait=True)

        if self.output_format == 'tiff':
            for target, asm in assemblers.items():
                if hasattr(asm, 'output_path'):
                    print(f"[tiff] {target} assembly complete: {', '.join(asm.output_path + '_' + str(c) for c in range(asm.output_channel))}")

        if self.output_format == 'zarr':
            for target, asm in assemblers.items():
                if hasattr(asm, 'zarr_path'):
                    print(f"[zarr] {target} assembly complete: {asm.zarr_path}")

        print(f"Done! Output saved to: {self.dest}")


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Microscopy inference + assembly')
    parser.add_argument('--env', type=str, default=None, help='Environment name from env.json (e.g. GHCL10, runpod, docker, ...)')
    parser.add_argument('--config', type=str, required=True, help='Config file name (without .yaml)')
    parser.add_argument('--option', type=str, default=None, help='Option/dataset name in config (optional, overrides DEFAULT)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    # CLI overrides for config values
    parser.add_argument('--input_image_filename', type=str, default=None, help='Input image filename (overrides config)')
    parser.add_argument('--output_dir_name', type=str, default=None, help='Output directory name (overrides config)')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path relative to SOURCE (overrides config)')
    parser.add_argument('--epoch', type=int, default=None, help='Model epoch (overrides config)')
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = InferencePipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

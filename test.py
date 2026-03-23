import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff
from omegaconf import OmegaConf
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
    """Layered OmegaConf config: base → scale → env → override → cli."""

    def __init__(self, scale=None, env=None, override=None, cli_overrides=None):
        # 1. Load base config
        cfg = OmegaConf.load('cfg/base.yaml')

        # 2. Merge scale config
        scale_path = f'cfg/scale/{scale}.yaml'
        if not os.path.exists(scale_path):
            raise FileNotFoundError(f"Scale config not found: {scale_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(scale_path))
        z_ratio = int(scale.replace('x', ''))
        cfg = OmegaConf.merge(cfg, OmegaConf.create({'scale': z_ratio}))

        # 3. Merge env config
        if not os.path.exists('cfg/env.yaml'):
            raise FileNotFoundError("cfg/env.yaml not found. Please read README.md to create this file!")
        env_all = OmegaConf.load('cfg/env.yaml')
        if env is None or env not in env_all:
            available = list(env_all.keys())
            raise ValueError(f"Environment '{env}' not found in env.yaml. Available: {available}")
        cfg = OmegaConf.merge(cfg, OmegaConf.create({'env': env_all[env]}))

        # 4. Merge override config
        if override:
            override_path = f'cfg/{override}.yaml'
            if not os.path.exists(override_path):
                raise FileNotFoundError(f"Override config not found: {override_path}")
            cfg = OmegaConf.merge(cfg, OmegaConf.load(override_path))

        # 5. Merge CLI overrides (e.g. model.epoch=2000 model.fp16=true)
        if cli_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))

        # Resolve all interpolations
        OmegaConf.resolve(cfg)

        if cfg.paths.version is None:
            cfg.paths.version = ''

        # Validate required paths
        required = ['input_img_relpath', 'ckpt_relpath', 'output_dir_name']
        missing = [k for k in required if OmegaConf.select(cfg, f'paths.{k}') is None]
        if missing:
            raise ValueError(f"Missing required fields in paths: {missing}")

        # Build runtime paths with os.path.join (avoids double-slash issues)
        cfg.runtime = OmegaConf.create({
            'input_img_path': os.path.join(cfg.env.DATASET, cfg.paths.input_img_relpath),
            'ckpt_root_path': os.path.join(cfg.env.MODEL, cfg.paths.ckpt_relpath, cfg.paths.version),
            'output_dir': os.path.join(cfg.env.RESULT, cfg.paths.output_dir_name),
        })

        self.cfg = cfg


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

    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.gpu = (self.device.type == 'cuda')

        self.model = self._load_model()
        self.upsample = torch.nn.Upsample(size=list(cfg.patch.upsample_size), mode='trilinear')

        if self.gpu:
            self.upsample = self.upsample.to(self.device)

        # Create flat dict from model config for ModelProcesser compatibility
        model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
        model_kwargs['model_type'] = model_kwargs.pop('type')
        self.model_proc = ModelProcesser(
            model_kwargs,
            self.model,
            gpu=self.gpu,
            augmentation=cfg.model.tta_mode,
            fp16=cfg.model.fp16,
            device=self.device
        )

    def _load_model(self):
        ckpt_root_path = self.cfg.runtime.ckpt_root_path
        epoch = self.cfg.model.epoch
        model_type = self.cfg.model.type
        print(f"Loading model from checkpoint, epoch: {epoch}, model type: {model_type}")

        if model_type == 'AE':
            model = self._load_ae(ckpt_root_path, epoch)
        elif model_type == 'GAN':
            model = self._load_gan(ckpt_root_path, epoch)
        elif model_type == 'VQQ2':
            model = self._load_vqq2(ckpt_root_path, epoch)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if self.gpu:
            model = model.to(self.device) if hasattr(model, 'to') else model
        for p in model.parameters():
            p.requires_grad = False
        if self.cfg.model.fp16:
            model = model.half()

        return model

    def _load_ae(self, ckpt_root_path, epoch):
        args = read_json_to_args(os.path.join(ckpt_root_path, "0.json"))
        model_module = import_model(ckpt_root_path, model_name=args.models)
        model = model_module.GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
        model = load_pth(model, root=ckpt_root_path, epoch=epoch,
                         model_names=['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv'])
        return model

    def _load_gan(self, ckpt_root_path, epoch):
        model_path = os.path.join(ckpt_root_path, f"net_g_model_epoch_{epoch}.pth")
        return torch.load(model_path, map_location='cpu', weights_only=False)

    def _load_vqq2(self, ckpt_root_path, epoch):
        return VQQModel(ckpt_root_path, epoch)


# =============================================================================
# ImageLoader
# =============================================================================
class ImageLoader:
    """Handles image loading and preprocessing."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.normalizer = DataNormalization(backward_type='float32')
        self.img = None

    def load(self):
        image_path = self.cfg.runtime.input_img_path
        print("=" * 50)
        print(f"Loading image from: {image_path}")
        img = tiff.imread(image_path) # (Z, Y, X) # (237, 2763, 2756)
        self.img = self.normalizer.forward_normalization(
            img, self.cfg.preprocess.norm_method[0], self.cfg.preprocess.trd[0]
        ) # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)
        return self.img

    def apply_percentile_norm(self):
        if self.img is None:
            raise ValueError("Image not loaded. Call load() first.")

        norm_pct = self.cfg.preprocess.norm_percentile
        if norm_pct is not False:
            p0, p1 = norm_pct
            if not isinstance(p0, (int, float)):
                p0, p1 = 0.5, 99.5

            xmin = np.percentile(self.img[..., ::2, ::2].flatten(), p0)
            xmax = np.percentile(self.img[..., ::2, ::2].flatten(), p1)
            print(f"Percentile clipping: [{xmin:.2f}, {xmax:.2f}]")

            self.img = torch.clamp(self.img, xmin, xmax)
            self.img = (self.img - xmin) / (xmax - xmin)
            self.img = self.img * 2 - 1

        return self.img # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)

    def cropROI(self, zrange, yrange, xrange):
        """Crop image to specified range (used when testwhole=False)."""
        if self.img is None:
            raise ValueError("Image not loaded. Call load() first.")
        self.img = self.img[:, :, zrange[0]:zrange[1], yrange[0]:yrange[1], xrange[0]:xrange[1]]
        print(f"Cropped to {self.img.shape}")
        return self.img

    def pad(self, crop_margin=None):
        """
        Pad image for patch-based inference.

        Args:
            crop_margin: (Cz, Cy, Cx) pixels cropped from each side during assembly.
                         If provided, pads by C on both sides to preserve original size.
        """
        print("=" * 50)
        if self.img is None:
            raise ValueError("Image not loaded. Call load() first.")

        _, _, Dz, Dy, Dx = self.img.shape # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)
        print(f"Original size (Z, Y, X): ({Dz}, {Dy}, {Dx})")

        # First, pad by crop margin (C) on both sides to preserve original size after assembly
        if crop_margin is not None:
            Cz, Cy, Cx = crop_margin
            z_ratio = self.cfg.scale
            self.img = F.pad(self.img, (Cx, Cx, Cy, Cy, Cz//z_ratio, Cz//z_ratio), mode='constant', value=self.img.mean())
            _, _, Dz, Dy, Dx = self.img.shape # (1, 1, Z, Y, X) # (1, 1, 245, 2795, 2788)
            print(f"After C padding size (Z, Y, X): {self.img.shape}")

        zstep = int(eval(str(self.cfg.grid.step.z)))
        ystep = int(eval(str(self.cfg.grid.step.y)))
        xstep = int(eval(str(self.cfg.grid.step.x)))
        # Then pad to make divisible by patch shape
        Nz = ((Dz // zstep) + 1) * zstep
        Ny = ((Dy // ystep) + 1) * ystep
        Nx = ((Dx // xstep) + 1) * xstep

        Pz, Py, Px = Nz - Dz, Ny - Dy, Nx - Dx
        self.img = F.pad(self.img, (0, Px, 0, Py, 0, Pz), mode='constant', value=self.img.mean())
        print(f"After stride padding size (Z, Y, X): {self.img.shape}")

        return self.img # (1, 1, Z, Y, X) # (1, 1, 260, 2912, 2912)


# =============================================================================
# PatchProcessor
# =============================================================================
class PatchProcessor:
    """Handles inference on patches."""

    def __init__(self, cfg, model_proc, upsample, device='cpu'):
        self.model_proc = model_proc
        self.upsample = upsample
        self.tta_method = list(cfg.model.tta_method) * cfg.model.num_mc
        self.mc_threshold = cfg.model.mc_threshold
        self.compute_xystd = 'xystd' in cfg.output.save
        self.fp16 = cfg.model.fp16
        self.device = torch.device(device)
        self.gpu = (self.device.type == 'cuda')

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
                _, Xup, outall, _ = self.model_proc.get_model_result(patch, self.tta_method, ii=ii)
        else:
            # Xup: (Z, C, Y, X) # (256, 1, 256, 256)
            # outall: (Z, C, Y, X, aug) # (256, 1, 256, 256, 2)
            _, Xup, outall, _ = self.model_proc.get_model_result(patch, self.tta_method, ii=ii)

        outall = outall.numpy().astype(np.float32) # (Z, C, Y, X, aug) # (256, 1, 256, 256, 2)
        Xup = Xup.numpy().astype(np.float32) # (Z, C, Y, X) # (256, 1, 256, 256)

        # Match output range to input range per channel
        for c in range(outall.shape[1]):
            omin, omax = outall[:, c, :, :, :].min(), outall[:, c, :, :, :].max()
            xmin, xmax = Xup[:, c, :, :].min(), Xup[:, c, :, :].max()
            outall[:, c, :, :, :] = (outall[:, c, :, :, :] - omin) / (omax - omin + 1e-6) * (xmax - xmin) + xmin
        # Xup: (Z, C, Y, X) # (256, 1, 256, 256)
        # outall: (Z, C, Y, X, aug) # (256, 1, 256, 256, 2)
        outstd = None
        if self.compute_xystd:
            outstd = ((outall > self.mc_threshold) / 1).std(axis=-1)  # binary MC std (Z, C, Y, X)
        return outall, Xup, outstd


# =============================================================================
# PatchAssembler
# =============================================================================
class PatchAssembler:

    def __init__(self, cfg, target_name, zrange, xrange, yrange):
        self.Cz, self.Cy, self.Cx = cfg.assemble.C
        self.Sz, self.Sy, self.Sx = cfg.assemble.S
        self.weight_shape = list(cfg.assemble.weight_shape)
        self.output_format = cfg.output.output_format
        self.target_name = target_name
        self.output_datatype = 'uint8' if target_name == 'xystd' else cfg.output.output_datatype
        self.output_channel = cfg.output.output_channel
        self.zrange = zrange
        self.xrange = xrange
        self.yrange = yrange

        self.patches = {}
        self.last_block = None
        self.current_x_position = 0

        # Setup output
        self.output_path = os.path.join(cfg.runtime.output_dir, target_name + '_assemble')
        if self.output_format == 'tiff':
            self._init_tiff_dirs()
        elif self.output_format == 'zarr':
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

        save_dtype = np.dtype(self.output_datatype)

        # TCZYX 5D for OME-Zarr: store X as zarr Z-axis (side view)
        shape_5d = (1, self.output_channel, fX, fZ, fY)
        chunks_5d = (1, 1, 20, 512, 512)

        zarr_path = self.output_path + '.zarr'
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.open_group(store, mode="w")
        self.zarr_out = root.create_dataset(
            "0", shape=shape_5d, chunks=chunks_5d, dtype=save_dtype, fill_value=0
        )
        self.zarr_path = zarr_path
        print(f"[zarr] Created store: {zarr_path}, shape TCZYX={shape_5d}")

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

    def _convert_dtype(self, data):
        """Convert float32 data to target dtype."""
        if self.target_name == 'xystd':
            dmin, dmax = data.min(), data.max()
            data = (data - dmin) / (dmax - dmin + 1e-6) * 255
            return data.astype(np.uint8)
        if self.output_datatype == 'uint8':
            return np.clip((data + 1) * 127.5, 0, 255).astype(np.uint8)
        elif self.output_datatype == 'uint16':
            return np.clip((data + 1) * 32767.5, 0, 65535).astype(np.uint16)
        return data.astype(np.float32)

    def _create_tapered_weight(self, Sz, Sy, Sx, nz, ny, nx, size, edge_size: int = 64) -> np.ndarray:
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
                # Edge flags: 
                # 0 = first patch, -1 = last patch
                nz_flag = -1 if nz == nz_count - 1 else nz
                nx_flag = -1 if nx == nx_count - 1 else nx
                ny_flag = -1 if ny == ny_count - 1 else ny

                # Get patch and crop margins from all sides
                patch = patches[(nz, ny)]  # (C, Z, X, Y) # (1, 256, 256, 256)
                cropped = patch[:, Cz:-Cz, Cy:-Cy, Cx:-Cx] # (C, Z, X, Y) # (1, 224, 224, 224)

                w = self._create_tapered_weight(Sz, Sy, Sx, nz_flag, ny_flag, nx_flag,
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


# =============================================================================
# InferencePipeline
# =============================================================================
class InferencePipeline:
    """patch inference + assembly pipeline."""

    def __init__(self, args):
        self.args = args
        config = Config(scale=args.scale, env=args.env, override=args.override, cli_overrides=args.cli_overrides)
        self.cfg = config.cfg
        print("===== Resolved Config =====")
        print(OmegaConf.to_yaml(self.cfg))
        print("=" * 50)

        # Validate MC config
        if self.cfg.model.num_mc > 1 and self.cfg.model.mc_threshold is None:
            raise ValueError("num_mc > 1 requires mc_threshold to be set (note that mc_threshold is float, range is [-1, 1])")

        # Setup output directory
        self.dest = self.cfg.runtime.output_dir
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
            loader = ModelLoader(self.cfg, device=device)
            proc = PatchProcessor(
                self.cfg, loader.model_proc, loader.upsample, device=device
            )
            self.workers.append(proc)

        self.image_loader = ImageLoader(self.cfg)

    def _setup_output_dirs(self):
        os.makedirs(self.dest, exist_ok=True)
        # Save resolved config
        with open(os.path.join(self.dest, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    def _get_patch_grid(self, img_shape):
        _, _, z, y, x = img_shape

        zstep = int(eval(str(self.cfg.grid.step.z)))
        ystep = int(eval(str(self.cfg.grid.step.y)))
        xstep = int(eval(str(self.cfg.grid.step.x)))

        return range(0, z, zstep), range(0, y, ystep), range(0, x, xstep)

    def run(self):
        # Load and preprocess image
        self.image_loader.load() # (B, C, Z, Y, X) # (1, 1, 237, 2763, 2756)
        self.image_loader.apply_percentile_norm() # (B, C, Z, Y, X) # (1, 1, 237, 2763, 2756)
        patch_shape = list(self.cfg.patch.patch_shape)
        dz, dy, dx = patch_shape

        crop_margin = list(self.cfg.assemble.C)
        # When testwhole=False, crop image to specified range first
        if not self.cfg.grid.testwhole:
            zr = list(self.cfg.grid.roi.z)
            yr = list(self.cfg.grid.roi.y)
            xr = list(self.cfg.grid.roi.x)
            self.image_loader.cropROI(zr, yr, xr)
            crop_margin = None

        self.image_loader.pad(crop_margin=crop_margin) # (B, C, Z, Y, X) # (1, 1, 260, 2912, 2912)
        x0 = [self.image_loader.img]

        # Get patch grid
        zrange, yrange, xrange = self._get_patch_grid(self.image_loader.img.shape)
        total_patches = len(zrange) * len(yrange) * len(xrange)
        print("=" * 50)
        print(f"Processing {len(zrange)}x{len(yrange)}x{len(xrange)} = {total_patches} patches")

        # Create one assembler per save target
        assemblers = {}
        for target in self.cfg.output.save:
            assemblers[target] = PatchAssembler(
                cfg=self.cfg,
                target_name=target,
                zrange=zrange, xrange=xrange, yrange=yrange
            )

        print('xrange:', xrange)
        print('zrange:', zrange)
        print('yrange:', yrange)
        print("=" * 50)
        print("Starting inference and assembly...")

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
                        outall, Xup, outstd = fut.result()
                        # outall: (Z, C, Y, X, aug) # (256, 1, 256, 256, 2)
                        # Xup: (Z, C, Y, X) # (256, 1, 256, 256)
                        # outstd: (Z, C, Y, X) # (256, 1, 256, 256) MC std
                        if 'xy' in assemblers:
                            assemblers['xy'].store_patch(nz, ny, outall, is_outall=True)
                        if 'ori' in assemblers:
                            assemblers['ori'].store_patch(nz, ny, Xup, is_outall=False)
                        if 'xystd' in assemblers:
                            assemblers['xystd'].store_patch(nz, ny, outstd, is_outall=False)
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

        if self.cfg.output.output_format == 'tiff':
            for target, asm in assemblers.items():
                if hasattr(asm, 'output_path'):
                    print(f"[tiff] {target} assembly complete: {', '.join(asm.output_path + '_' + str(c) for c in range(asm.output_channel))}")

        if self.cfg.output.output_format == 'zarr':
            for target, asm in assemblers.items():
                if hasattr(asm, 'zarr_path'):
                    print(f"[zarr] {target} assembly complete: {asm.zarr_path}")

        print(f"Done! Output saved to: {self.dest}")


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Microscopy inference + assembly')
    parser.add_argument('--env', type=str, required=True, help='Environment name from env.yaml (e.g. Docker, GHCL00, ...)')
    parser.add_argument('--scale', type=str, required=True, choices=['2x', '4x', '8x'], help='Scale config name from cfg/scale/')
    parser.add_argument('--override', type=str, default=None, help='Override config file name in `cfg/` (without `.yaml`)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')

    args, cli_overrides = parser.parse_known_args()
    args.cli_overrides = cli_overrides
    return args


def main():
    args = parse_args()
    pipeline = InferencePipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

import argparse
import os
import shutil
import time
import numpy as np
import math
import torch
import torch.nn.functional as F
import tifffile as tiff
from omegaconf import OmegaConf
import zarr
from ome_zarr.writer import write_multiscales_metadata
from tqdm import tqdm
from collections import deque
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
    """Layered OmegaConf config: base → scale → env → overrisde → cli."""

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

        # Validate required paths
        required = ['input_img_relpath', 'ckpt_relpath', 'output_dir_name']
        missing = [k for k in required if OmegaConf.select(cfg, f'paths.{k}') is None]
        if missing:
            raise ValueError(f"Missing required fields in paths: {missing}")

        # Build runtime paths with os.path.join (avoids double-slash issues)
        cfg.runtime = OmegaConf.create({
            'input_img_path': os.path.join(cfg.env.DATASET, cfg.paths.input_img_relpath),
            'ckpt_root_path': os.path.join(cfg.env.MODEL, cfg.paths.ckpt_relpath),
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
        self.upsample = torch.nn.Upsample(size=list(self.cfg.patch.upsample_size), mode='trilinear')

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
        return torch.load(model_path, map_location='cpu')

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

    def load_3d_img(self):
        image_path = self.cfg.runtime.input_img_path
        print("=" * 50)
        print(f"Loading image from: {image_path}")
        self.img = tiff.imread(image_path) # (Z, Y, X) # (237, 2763, 2756)
        return self.img

    def apply_forward_norm(self):
        if self.img is None:
            raise ValueError("Image not loaded. Call load_3d_img() first.")
        print("Applying forward normalization...")
        
        self.img = self.normalizer.forward_normalization(
            self.img, self.cfg.preprocess.norm_method[0], self.cfg.preprocess.trd[0]
        ) # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)
        return self.img

    def apply_percentile_norm(self):
        if self.img is None:
            raise ValueError("Image not loaded. Call load_3d_img() first.")
        print("Applying percentile normalization...")

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

    def crop_roi(self, zrange, yrange, xrange):
        """Crop image to specified range (used when testwhole=False)."""
        if self.img is None:
            raise ValueError("Image not loaded. Call load_3d_img() first.")
        self.img = self.img[:, :, zrange[0]:zrange[1], yrange[0]:yrange[1], xrange[0]:xrange[1]]
        print(f"Cropped to {self.img.shape}")
        return self.img

    def pad(self, patch_shape, crop_margin=None):
        """
        Pad image for patch-based inference.

        Args:
            crop_margin: (Cz, Cy, Cx) pixels cropped from each side during assembly.
                        If provided, pads by C on both sides to preserve original size.
        """
        print("=" * 50)
        if self.img is None:
            raise ValueError("Image not loaded. Call load_3d_img() first.")
        dz, dy, dx = patch_shape
        _, _, Dz, Dy, Dx = self.img.shape # (1, 1, Z, Y, X)
        print(f"Original size (Z, Y, X): ({Dz}, {Dy}, {Dx})")

        # First, pad by crop margin (C) on both sides to preserve original size after assembly
        if crop_margin is not None:
            Cz, Cy, Cx = crop_margin
            z_ratio = self.cfg.scale
            self.img = F.pad(self.img, (Cx, Cx, Cy, Cy, Cz//z_ratio, Cz//z_ratio), mode='constant', value=self.img.mean())
            _, _, Dz, Dy, Dx = self.img.shape # (1, 1, Z, Y, X)
            print(f"After C padding size (Z, Y, X): {(self.img.shape[2], self.img.shape[3], self.img.shape[4])}")

        zstep = int(eval(str(self.cfg.grid.step.z)))
        ystep = int(eval(str(self.cfg.grid.step.y)))
        xstep = int(eval(str(self.cfg.grid.step.x)))

        def pad_for_stride(D, p, s):
            n = max(1, math.ceil((D - p) / s) + 1)
            return (n - 1) * s + p

        Nz = pad_for_stride(Dz, dz, zstep)
        Ny = pad_for_stride(Dy, dy, ystep)
        Nx = pad_for_stride(Dx, dx, xstep)
        Pz, Py, Px = Nz - Dz, Ny - Dy, Nx - Dx
        self.img = F.pad(self.img, (0, Px, 0, Py, 0, Pz), mode='constant', value=self.img.mean())
        print(f"After stride padding size (Z, Y, X): {(self.img.shape[2], self.img.shape[3], self.img.shape[4])}")

        return self.img # (1, 1, Z, Y, X)


# =============================================================================
# PatchProcessor
# =============================================================================
class PatchProcessor:
    """Handles inference on patches."""

    def __init__(self, cfg, model_proc, upsample, resample, device='cpu'):
        self.model_proc = model_proc
        self.upsample = upsample
        self.resample = resample
        self.tta_method = list(cfg.model.tta_method) * cfg.model.num_mc
        self.mc_threshold = cfg.model.mc_threshold
        self.compute_xystd = 'xystd' in cfg.output.save
        self.fp16 = cfg.model.fp16
        self.device = torch.device(device)
        self.gpu = (self.device.type == 'cuda')

    def run(self, imgs, start_index, patch_shape, resample=None, ii=None):
        """Run inference on a batch of patches.

        Args:
            start_index: list of [iz, iy, ix] starts, length B.

        Returns:
            list of B tuples (xy_patch, Xup, None)
        """
        B = len(start_index)
        pz, py, px = patch_shape

        per_channel = []
        for img in imgs:
            patch = [img[:, :, si[0]:si[0]+pz, si[1]:si[1]+py, si[2]:si[2]+px] for si in start_index]
            if self.resample:
                print(f"Applying resampling with factor {self.resample} to input patches...")
                patch = [p[:, :, ::int(self.resample), :, :] for p in patch]
            batch_patch = torch.cat(patch, dim=0)
            batch_patch = batch_patch.to(self.device)
            per_channel.append(self.upsample(batch_patch))

        batch_input = torch.cat(per_channel, dim=1)

        if self.fp16 and self.gpu:
            batch_input = batch_input.half() # (B, C, Z, Y, X)

        if self.compute_xystd:
            XupX, Xup, outstd, _ = self.model_proc.get_vqq_out_batch_mc(
                batch_input, self.tta_method, self.mc_threshold, ii=ii)
            outstd = outstd.numpy().astype(np.float32)
        else:
            XupX, Xup, _ = self.model_proc.get_vqq_out_batch(
                batch_input, self.tta_method, ii=ii)
            outstd = None
        # XupX: (B, Z, C, X, Y) — already normalized mean, CPU
        # Xup:  (B, Z, C, X, Y) — CPU
        # outstd: (B, Z, C, X, Y) or None

        XupX = XupX.numpy().astype(np.float32)
        Xup = Xup.numpy().astype(np.float32)

        results = []
        for b in range(B):
            results.append((XupX[b], Xup[b], None if outstd is None else outstd[b]))
        return results


# =============================================================================
# PatchAssembler
# =============================================================================
class PatchAssembler:
    """Per-column assembler with no cross-column blending.

    Each column writes its full block (including BOTH leading and trailing Sx
    overlap regions) to its own storage location. X-axis taper-blending across
    columns is deferred to an external post-processing step.

    Post-processing contract:
        Adjacent columns' overlap regions are stored as *complementary tapered*
        values (column N's trailing Sx tapers 1→0, column N+1's leading Sx
        tapers 0→1, weights sum to 1). The post-processor must ADD them, not
        pick one, to recover full intensity.
    """

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
        self._weight_cache = {}

        # Setup output
        self.output_path = os.path.join(cfg.runtime.output_dir, target_name + '_assemble')
        if self.output_format == 'tiff':
            self._init_tiff_dirs()
        elif self.output_format == 'zarr':
            self._init_zarr_store()

    def _init_tiff_dirs(self):
        nx_count = len(self.xrange)
        for c in range(self.output_channel):
            d = self.output_path + '_' + str(c)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
            for nx in range(nx_count):
                os.makedirs(os.path.join(d, f'col_{nx:03d}'), exist_ok=True)

    def _init_zarr_store(self):
        Wz, Wy, Wx = self.weight_shape
        nz = len(self.zrange)
        ny = len(self.yrange)
        nx = len(self.xrange)
        fZ = int(Wz * nz - self.Sz * (nz - 1))
        fY = int(Wy * ny - self.Sy * (ny - 1))
        # No cross-column X blending: each column occupies a full Wx slab,
        # adjacent columns' overlaps live side-by-side and are merged downstream.
        fX = int(Wx * nx)

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
        # Force creation of 0/.zattrs — Avivator fetches it and 404s if missing
        self.zarr_out.attrs.put({})
        self._write_ome_metadata(root)
        self.zarr_path = zarr_path
        print(f"[zarr] Created store: {zarr_path}, shape TCZYX={shape_5d}")

    def _write_ome_metadata(self, root):
        """Write OME-NGFF v0.4 metadata so viewers (Avivator, napari, vizarr)
        can read the zarr. Axes order matches array shape (T, C, Z, Y, X)."""
        write_multiscales_metadata(
            group=root,
            datasets=[{
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}
                ],
            }],
            axes=[
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "pixel"},
                {"name": "y", "type": "space", "unit": "pixel"},
                {"name": "x", "type": "space", "unit": "pixel"},
            ],
            name=f"{self.target_name}_sideview_YZ",
        )

        # omero block — default contrast window for viewers
        if self.target_name == 'xystd' or self.output_datatype == 'uint8':
            wmin, wmax = 0.0, 255.0
        elif self.output_datatype == 'uint16':
            wmin, wmax = 0.0, 65535.0
        else:  # float32, model output is roughly [-1, 1]
            wmin, wmax = -1.0, 1.0
        root.attrs["omero"] = {
            "channels": [{
                "label": f"ch{c}",
                "color": "FFFFFF",
                "window": {"min": wmin, "max": wmax, "start": wmin, "end": wmax},
                "active": True,
            } for c in range(self.output_channel)],
            "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "greyscale"},
        }

    def _write_column_tiff(self, one_zy_block, nx):
        """Write full column (incl. both Sx overlaps) under per-column subdir."""
        for x in range(one_zy_block.shape[1]):
            for c in range(one_zy_block.shape[0]):
                slice_data = one_zy_block[c, x, :, :]
                slice_data = self._convert_dtype(slice_data)
                tiff.imwrite(
                    os.path.join(self.output_path + '_' + str(c),
                                 f'col_{nx:03d}',
                                 f'slice_x_{x:04d}.tif'),
                    slice_data
                )

    def _write_column_zarr(self, one_zy_block, nx):
        """Write full column block at [nx*Wx : (nx+1)*Wx] in zarr store."""
        block = self._convert_dtype(one_zy_block)
        Wx = self.weight_shape[2]
        x_start = nx * Wx
        x_end = x_start + one_zy_block.shape[1]
        self.zarr_out[0, :, x_start:x_end, :, :] = block

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

    def _create_tapered_weight(self, nz_flag, ny_flag, nx_flag) -> np.ndarray:
        """Cached tapered weight. Boundary states per axis collapse to 3 cases
        (first / last / middle), so at most 27 weight arrays are materialized.
        """
        # Normalize middle indices (anything not 0 or -1) to a single key
        def _state(f):
            if f == 0 or f == -1:
                return f
            return 'M'
        key = (_state(nz_flag), _state(ny_flag), _state(nx_flag))
        cached = self._weight_cache.get(key)
        if cached is not None:
            return cached

        Sz, Sy, Sx = self.Sz, self.Sy, self.Sx
        weight = np.ones(self.weight_shape, dtype=np.float32)
        taper_Sz = np.linspace(0, 1, Sz, dtype=np.float32)
        taper_Sy = np.linspace(0, 1, Sy, dtype=np.float32)
        taper_Sx = np.linspace(0, 1, Sx, dtype=np.float32)
        nz_k, ny_k, nx_k = key
        if nz_k != 0:
            weight[:Sz, :, :] *= taper_Sz.reshape(-1, 1, 1)
        if nz_k != -1:
            weight[-Sz:, :, :] *= taper_Sz[::-1].reshape(-1, 1, 1)
        if ny_k != 0:
            weight[:, :Sy, :] *= taper_Sy.reshape(1, -1, 1)
        if ny_k != -1:
            weight[:, -Sy:, :] *= taper_Sy[::-1].reshape(1, -1, 1)
        if nx_k != 0:
            weight[:, :, :Sx] *= taper_Sx.reshape(1, 1, -1)
        if nx_k != -1:
            weight[:, :, -Sx:] *= taper_Sx[::-1].reshape(1, 1, -1)

        weight.setflags(write=False)
        self._weight_cache[key] = weight
        return weight
    
    def store_patch(self, nz, ny, data, aug_dim_avg=False):
        """Store a patch for the current X loop.

        Args:
            nz, ny: grid indices within current X loop
            data: outall (C, Z, Y, X, aug) if is_outall, else Xup (C, Z, Y, X)
            is_outall: True if data has augmentation dimension to average
        """
        if aug_dim_avg:
            data = data.mean(axis=-1)  # average augmentations => (C, Z, Y, X, aug) -> (C, Z, Y, X)
        patch = data.astype(np.float32)  # (C, Z, Y, X)
        self.patches[(nz, ny)] = patch

    def assemble_column(self, nx, patches=None, write_executor=None):
        """Assemble Y/Z-blended column at index nx and write the full block.

        Y/Z blending happens in-RAM as before. X-axis blending is NOT performed
        here — the full column (incl. both Sx overlaps) is written as-is, and
        adjacent columns are merged in a separate post-processing pass.

        Columns are now fully independent — multiple `assemble_column` calls
        can run concurrently with no shared mutable state.
        """
        if patches is None:
            patches = self.patches
        Cz, Cy, Cx = self.Cz, self.Cy, self.Cx
        Sz, Sy = self.Sz, self.Sy
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

                w = self._create_tapered_weight(nz_flag, ny_flag, nx_flag)
                cropped = cropped * w  # broadcast (Z,Y,X) over C → (C,Z,Y,X)

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

        one_zy_block = np.concatenate(one_zy_block, axis=2)  # (C, X=Wx, Z_total, Y_total)

        # X-blending intentionally skipped — see class docstring.
        def _write2disk():
            if self.output_format == 'tiff':
                self._write_column_tiff(one_zy_block, nx)
            elif self.output_format == 'zarr':
                self._write_column_zarr(one_zy_block, nx)

        if write_executor is not None:
            write_executor.submit(_write2disk)
        else:
            _write2disk()


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
        if 'xystd' in self.cfg.output.save and self.cfg.model.mc_threshold is None:
            raise ValueError("output.save contains 'xystd' but model.mc_threshold is not set")

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
                self.cfg, loader.model_proc, loader.upsample, resample=self.cfg.patch.resample, device=device
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
        pz, py, px = self.cfg.patch.patch_shape
        zstep = int(eval(str(self.cfg.grid.step.z)))
        ystep = int(eval(str(self.cfg.grid.step.y)))
        xstep = int(eval(str(self.cfg.grid.step.x)))

        zend = (z - pz) + zstep
        yend = (y - py) + ystep
        xend = (x - px) + xstep

        return range(0, int(zend), zstep), range(0, int(yend), ystep), range(0, int(xend), xstep)

    def run(self):
        # Load and preprocess image
        self.image_loader.load_3d_img() # (Z, Y, X) # (237, 2763, 2756)
        self.image_loader.apply_forward_norm() # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)
        self.image_loader.apply_percentile_norm() # (1, 1, Z, Y, X) # (1, 1, 237, 2763, 2756)

        crop_margin = self.cfg.assemble.C
        # When testwhole=False, crop image to specified range first
        if not self.cfg.grid.testwhole:
            zr = list(self.cfg.grid.roi.z)
            yr = list(self.cfg.grid.roi.y)
            xr = list(self.cfg.grid.roi.x)
            self.image_loader.crop_roi(zr, yr, xr)
            crop_margin = None

        patch_shape = self.cfg.patch.patch_shape
        self.image_loader.pad(patch_shape=patch_shape, crop_margin=crop_margin) # (B, C, Z, Y, X) # (1, 1, 260, 2912, 2912)
        imgs = [self.image_loader.img]
        
        # Get patch grid
        zrange, yrange, xrange = self._get_patch_grid(self.image_loader.img.shape)
        total_patches = len(zrange) * len(yrange) * len(xrange)
        print("=" * 50)
        print('zrange:', zrange)
        print('yrange:', yrange)
        print('xrange:', xrange)
        print(f"Processing {len(zrange)}x{len(yrange)}x{len(xrange)} = {total_patches} patches")

        # Create one assembler per save target
        assemblers = {}
        for target in self.cfg.output.save:
            assemblers[target] = PatchAssembler(
                cfg=self.cfg,
                target_name=target,
                zrange=zrange, xrange=xrange, yrange=yrange
            )
        # Build list of (nz, ny, iz, iy) for each x-column
        column_patches = []
        for nz, iz in enumerate(zrange):
            for ny, iy in enumerate(yrange):
                column_patches.append((nz, ny, iz, iy))

        print("=" * 50)
        print("Starting inference and assembly...")

        num_workers = len(self.workers)
        batch_size = int(OmegaConf.select(self.cfg, 'model.batch_size') or 1)
        assemble_executor = ThreadPoolExecutor(max_workers=1)

        MAX_ASSEMBLE_QUEUE_LENGTH = 4
        assemble_futures = deque()
        backpressure_count = 0
        backpressure_total_s = 0.0

        with tqdm(total=total_patches, desc="Processing") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as infer_executor:
                for nx, ix in enumerate(xrange):
                    # Group patches of this column into chunks of `batch_size`
                    batch_patches = [column_patches[i:i+batch_size]
                              for i in range(0, len(column_patches), batch_size)]
                    futures = {}
                    for idx, batch_patch in enumerate(batch_patches):
                        gpu_id = idx % num_workers
                        start_index = [[iz, iy, ix] for (_, _, iz, iy) in batch_patch]
                        fut = infer_executor.submit(
                            self.workers[gpu_id].run,
                            imgs,
                            start_index=start_index,
                            patch_shape=patch_shape,
                        )
                        # map future -> list of (nz, ny) in chunk order
                        futures[fut] = [(nz, ny) for (nz, ny, _, _) in batch_patch]

                    # Collect results (order doesn't matter for store_patch)
                    for fut in as_completed(futures):
                        nz_ny_list = futures[fut]
                        for (nz, ny), (XupX, Xup, outstd) in zip(nz_ny_list, fut.result()):
                            # XupX: (Z, C, X, Y) — already normalized mean
                            # Xup:     (Z, C, X, Y)
                            # outstd:  (Z, C, X, Y) or None
                            if 'xy' in assemblers:
                                assemblers['xy'].store_patch(nz, ny, XupX)
                            if 'ori' in assemblers:
                                assemblers['ori'].store_patch(nz, ny, Xup)
                            if 'xystd' in assemblers:
                                assemblers['xystd'].store_patch(nz, ny, outstd)
                            pbar.update(1)

                    # Snapshot patches and reset — so next column can store immediately
                    patches_snapshots = {}
                    for target, asm in assemblers.items():
                        patches_snapshots[target] = asm.patches
                        asm.patches = {}  # fresh dict for next column

                    def _assemble(assemblers, snapshots, nx):
                        for target, asm in assemblers.items():
                            asm.assemble_column(nx, patches=snapshots[target],
                                                write_executor=None)

                    while assemble_futures and assemble_futures[0].done():
                        assemble_futures.popleft().result()

                    if len(assemble_futures) >= MAX_ASSEMBLE_QUEUE_LENGTH:
                        tqdm.write(
                            f"[backpressure] assemble queue full "
                            f"({len(assemble_futures)}/{MAX_ASSEMBLE_QUEUE_LENGTH}) "
                            f"at column nx={nx} — GPU stalled, waiting on oldest assemble"
                        )
                        t_block = time.time()
                        while len(assemble_futures) >= MAX_ASSEMBLE_QUEUE_LENGTH:
                            assemble_futures.popleft().result()
                        blocked_s = time.time() - t_block
                        tqdm.write(f"[backpressure] unblocked after {blocked_s:.2f}s")
                        backpressure_count += 1
                        backpressure_total_s += blocked_s

                    assemble_futures.append(assemble_executor.submit(
                        _assemble, assemblers, patches_snapshots, nx
                    ))

        if assemble_futures:
            t_drain = time.time()
            while assemble_futures:
                assemble_futures.popleft().result()
                print(f"[final drain] ({len(assemble_futures)}/{MAX_ASSEMBLE_QUEUE_LENGTH}) remaining")
            print(f"[final drain] all assembles done after {time.time()-t_drain:.2f}s")
        assemble_executor.shutdown(wait=True)

        if backpressure_count == 0:
            print("[backpressure] never triggered — assemble kept up with inference")
        else:
            print(
                f"[backpressure] triggered {backpressure_count} time(s), "
                f"total GPU stall: {backpressure_total_s:.2f}s "
                f"(avg {backpressure_total_s/backpressure_count:.2f}s/event). "
                f"Consider raising MAX_ASSEMBLE_QUEUE_LENGTH if RAM allows, or "
                f"max_workers if a single assemble is slow."
            )

        if self.cfg.output.output_format == 'tiff':
            for target, asm in assemblers.items():
                if hasattr(asm, 'output_path'):
                    print(f"[tiff] {target} assembly complete: {', '.join(asm.output_path + '_' + str(c) for c in range(asm.output_channel))}")

        if self.cfg.output.output_format == 'zarr':
            for target, asm in assemblers.items():
                if hasattr(asm, 'zarr_path'):
                    print(f"[zarr] {target} assembly complete: {asm.zarr_path}")

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Microscopy Patch Inference & Assembly')
    parser.add_argument('--env', type=str, required=True, help='Environment name from env.yaml (e.g. Docker, GHCL00, ...)')
    parser.add_argument('--scale', type=str, required=True, choices=['2x', '4x', '8x'], help='Scale config name from cfg/scale/')
    parser.add_argument('--override', type=str, default=None, help='Override config file name in `cfg/` (without `.yaml`)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')

    args, cli_overrides = parser.parse_known_args()
    args.cli_overrides = cli_overrides

    pipeline = InferencePipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

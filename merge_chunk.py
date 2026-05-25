import argparse
import math
import os
import shutil
import sys

import numpy as np
import tifffile as tiff
from omegaconf import OmegaConf
from tqdm import tqdm


def reverse_dtype(arr):
    """Encoded uint8/uint16 back to signed float in [-1, 1]."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 127.5 - 1.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 32767.5 - 1.0
    return arr.astype(np.float32)


def encode_dtype(arr_f32, target_dtype):
    """Signed float in [-1, 1] back to the target output dtype."""
    if target_dtype == np.uint8:
        return np.clip((arr_f32 + 1.0) * 127.5, 0, 255).astype(np.uint8)
    if target_dtype == np.uint16:
        return np.clip((arr_f32 + 1.0) * 32767.5, 0, 65535).astype(np.uint16)
    return arr_f32.astype(np.float32)


def merge_one(target_dir, out_dir, Wx, Sx, target_dtype):
    """Merge col_NNN/ subdirs under target_dir into a flat slice sequence in out_dir.

    Layout assumptions:
      * Each col_NNN/ holds Wx slices named slice_x_XXXX.tif (XXXX = local 0..Wx-1).
      * Adjacent columns share an Sx-wide overlap. Column N's local x corresponds
        to global x = N * (Wx - Sx) + local_x.
    """
    col_dirs = sorted(d for d in os.listdir(target_dir) if d.startswith('col_'))
    if not col_dirs:
        print(f"[skip] no col_NNN/ subdirs in {target_dir}")
        return

    n_cols = len(col_dirs)
    step_x = Wx - Sx
    final_X = n_cols * step_x + Sx

    os.makedirs(out_dir, exist_ok=True)

    for gx in tqdm(range(final_X), desc=os.path.basename(target_dir)):
        # Columns whose stored range [N*step_x, N*step_x + Wx) contains gx.
        n_max = min(n_cols - 1, gx // step_x)
        n_min = max(0, math.ceil((gx - Wx + 1) / step_x))

        contributors = []
        for n in range(n_min, n_max + 1):
            local_x = gx - n * step_x
            src = os.path.join(target_dir, col_dirs[n], f'slice_x_{local_x:04d}.tif')
            if os.path.exists(src):
                contributors.append(src)

        if not contributors:
            continue

        dst = os.path.join(out_dir, f'slice_x_{gx:04d}.tif')
        if len(contributors) == 1:
            # Non-overlap zone — taper weight is 1, the stored value is already
            # the final value. Just copy the file.
            shutil.copyfile(contributors[0], dst)
            continue

        # Overlap zone — sum in signed-float space, then re-encode. Tapers are
        # linearly complementary across the boundary so the sum recovers full
        # intensity. Doing this in uint8 space would clip and corrupt; we must
        # reverse the encoding first.
        accum = reverse_dtype(tiff.imread(contributors[0]))
        for src in contributors[1:]:
            accum += reverse_dtype(tiff.imread(src))
        tiff.imwrite(dst, encode_dtype(accum, target_dtype))


def main():
    parser = argparse.ArgumentParser(
        description='Merge per-column TIFF output from test.py via X-axis taper blending.'
    )
    parser.add_argument('output_dir',
                        help='Output dir produced by test.py (contains config.yaml).')
    parser.add_argument('--targets', nargs='*', default=None,
                        help='Targets to merge (e.g. xy ori). Default: auto-discover.')
    parser.add_argument('--out-suffix', default='merged',
                        help='Output dir suffix: {target}_{out-suffix}_{c}/ (default: merged).')
    args = parser.parse_args()

    cfg_path = os.path.join(args.output_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        sys.exit(f"config.yaml not found in {args.output_dir}")

    cfg = OmegaConf.load(cfg_path)
    Wx = int(cfg.assemble.weight_shape[2])
    Sx = int(cfg.assemble.S[2])
    output_channel = int(cfg.output.output_channel)
    target_dtype = np.dtype(str(cfg.output.output_datatype))

    if args.targets:
        targets = list(args.targets)
    else:
        targets = sorted({
            d.rsplit('_assemble_', 1)[0]
            for d in os.listdir(args.output_dir)
            if '_assemble_' in d
        })
        print(f"Auto-discovered targets: {targets}")

    if 'xystd' in targets:
        print("[warn] xystd uses per-slice min-max normalization at write time, "
              "which is not reversible — its overlap regions cannot be correctly "
              "blended by summation. Skipping xystd.")
        targets = [t for t in targets if t != 'xystd']

    print(f"Wx={Wx}, Sx={Sx}, step_x={Wx-Sx}, dtype={target_dtype}, channels={output_channel}")

    for target in targets:
        for c in range(output_channel):
            target_dir = os.path.join(args.output_dir, f'{target}_assemble_{c}')
            if not os.path.isdir(target_dir):
                continue
            out_dir = os.path.join(args.output_dir, f'{target}_{args.out_suffix}_{c}')
            merge_one(target_dir, out_dir, Wx, Sx, target_dtype)


if __name__ == '__main__':
    main()

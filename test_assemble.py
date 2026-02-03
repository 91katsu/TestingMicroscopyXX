import argparse
import glob
import json
import os
import shutil
import sys
import time

import threading
import queue
import ipdb
import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
import traceback

import networks
import models
from utils.data_utils import imagesc, DataNormalization
from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser

import zarr
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torch.utils.data import DataLoader, Dataset


def reverse_log(x):
    return np.power(10, x)


def create_tapered_weight(S0, S1, S2, nz, nx, ny, size, edge_size: int = 64) -> np.ndarray:
    """
    Create a 3D cube with linearly tapered edges in all directions.

    Args:
        size (int): Size of the cube (size x size x size)
        edge_size (int): Size of the tapered edge section

    Returns:
        np.ndarray: 3D array with tapered weights
    """
    # Create base cube filled with ones
    weight = np.ones(size)

    # Create linear taper from 0 to 1
    # taper = np.linspace(0, 1, edge_size)
    taper_S0 = np.linspace(0, 1, S0)
    taper_S1 = np.linspace(0, 1, S1)
    taper_S2 = np.linspace(0, 1, S2)

    # Z
    if nz != 0:
        weight[:S0, :, :] *= taper_S0.reshape(-1, 1, 1)
    if nz != -1:
        weight[-S0:, :, :] *= taper_S0[::-1].reshape(-1, 1, 1)

    # X
    if nx != 0:
        weight[:, :S1, :] *= taper_S1.reshape(1, -1, 1)
    if nx != -1:
        weight[:, -S1:, :] *= taper_S1[::-1].reshape(1, -1, 1)

    # Y
    if ny != 0:
        weight[:, :, :S2] *= taper_S2
    if ny != -1:
        weight[:, :, -S2:] *= taper_S2[::-1]

    return weight


class MicroTest:
    def __init__(self):
        # Init all args for data and model
        self.init_params()

        # Init model and upsample
        self.model, self.upsample = None, None
        # Use output_dtype from config if available, otherwise fall back to CLI arg
        self.save_image_datatype = self.kwargs.get('output_dtype', self.args.image_datatype)
        self.normalization = DataNormalization(backward_type=self.save_image_datatype)

    def process_config(self, config, option):
        return {**config['DEFAULT'], **config[option]}

    def init_params(self):
        # Init / Update Model Args
        self.args = self.update_args()
        # Init / Update Data Args
        # self.kwargs, self.kwargs = self.update_data_args('test/' + self.args.config + '.yaml')

        self.kwargs = self.process_config(yaml.safe_load(open('test/' + self.args.config + '.yaml')), self.args.option)

        # Overwrite image_path and dataset if provided via CLI
        if getattr(self.args, 'image_path', None) is not None:
            self.kwargs['image_path'] = self.args.image_path
        if getattr(self.args, 'dataset', None):
            self.kwargs['dataset'] = self.args.dataset


        # print("kwargs : ", self.kwargs)
        # print("data : ", self.kwargs)

    def update_args(self):
        parser = argparse.ArgumentParser()
        # projects
        parser.add_argument('--config', type=str, default="config", help='which config file')
        parser.add_argument('--option', type=str, default="Default", help='which dataset to use')
        parser.add_argument('--assemble_method', type=str, default='tiff', help='tiff or zarr method while assemble images')
        parser.add_argument('--image_datatype', type=str, default="float32")
        parser.add_argument('--augmentation', type=str, default="encode")
        parser.add_argument('--roi', type=str, default='')
        parser.add_argument('--targets', nargs='+', default=None, required=False, help="assign target to assemble")
        parser.add_argument('--reslice', action='store_true', help='reslice the original images')
        # overwriting args
        parser.add_argument('--image_path', nargs='+', type=str, default=None, help='List of image paths (overrides YAML)')
        parser.add_argument('--dataset', type=str, default=None, help='Dataset name (overrides YAML)')

        return parser.parse_args()

    def update_data_args(self, config_name):
        with open(config_name, 'r') as f:
            config = yaml.safe_load(f)

        kwargs = config.get(self.args.option, {})
        if not kwargs:
            raise ValueError(f"Option {self.args.option} not found in the configuration.")
        return config, kwargs

    def get_data(self, norm=True):
        # 我覺得這要改，有兩種情況，一她只要開圖存，不見得真的要paired data,二 他要inference

        kwargs = self.kwargs

        image_list_path = self.kwargs.get("image_list_path")  # if image path is a directory
        hbranch_path = self.kwargs.get("hbranch_path")

        x0 = []

        if kwargs.get("image_path"):
            image_path = [self.kwargs.get("root_path") + x for x in
                          self.kwargs.get("image_path", [])]  # if image path is a file
            print("image_path : ", image_path)
            for i in range(len(image_path)):
                img = tiff.imread(image_path[i])
                if norm:
                    img = self.normalization.forward_normalization(img, self.kwargs["norm_method"][i], self.kwargs['trd'][i])
                x0.append(img)

        # I'm not sure image list path is used for 2D images or need deal with many 3D images cube
        # new method with 2D loading images
        # assert 2D image is (X, Y) output will be [(Z, X, Y), (Z, X, Y)]
        elif kwargs.get("image_list_path"):
            image_list_path = [kwargs.get("root_path") + x for x in
                               kwargs.get("image_list_path")]  # if image path is a directory
            for num, i in enumerate(image_list_path):
                ids = sorted(os.listdir(image_list_path[num]))
                img = np.stack([tiff.imread(os.path.join(i, id)) for id in ids], 0)
                if norm:
                    img = self.normalization.forward_normalization(img, self.kwargs["norm_method"][num],
                                       self.kwargs['trd'][num])
                x0.append(img)

        elif hbranch_path:
            class HbranchDataset(Dataset):
                def __init__(self, folder):
                    # 搜尋資料夾下所有 npy 檔案，並根據檔名排序（你也可以根據需要修改排序規則）
                    self.files = sorted(glob.glob(os.path.join(folder, "*.npy")))
                    if len(self.files) == 0:
                        raise ValueError(f"No .npy files found in {folder}")

                def __len__(self):
                    return len(self.files)

                def __getitem__(self, idx):
                    # 載入 npy 檔案，可以視需要加入資料轉換（例如轉成 torch.Tensor）
                    return np.load(self.files[idx], allow_pickle=True), self.files[idx]

            hbranch_dataset = HbranchDataset(hbranch_path)
            # 這裡預設 batch_size 為 1，你可以根據需求調整 batch_size 與 shuffle 設定
            x0 = DataLoader(hbranch_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

        else:
            raise ValueError("No valid image path provided.")

        print("x0 shape : ", x0[0].shape)

        return x0

    def save_images(self, outpath, img, axis=None, norm_method=None, exp_trd=None, trd=None):
        """
        儲存影像的函數，接受完整的輸出路徑。

        參數：
        - outpath (str): 輸出的檔案路徑，可以包含目錄（例如 "abc/def/xxx.tif"）或僅為檔案名稱（例如 "xxx.tif"）。
        - img (np.ndarray): 要儲存的影像數據。
        - axis (tuple, optional): 影像轉置的軸。
        """
        directory = os.path.dirname(outpath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 處理影像
        # if self.args.reverselog:
        #     img = reverse_log(img)

        if norm_method:
            img = self.normalization.backward_normalization(img, norm_method, exp_trd, trd)

        if axis is not None:
            img = np.transpose(img, axis)

        # 儲存影像
        tiff.imwrite(self.kwargs['root_path'] + outpath, img)

    def show_or_save_assemble_microscopy(self, zrange, xrange, yrange, source, output_path="tmp.tif", show=True):
        """
        output_path == "" is to show only
        __future__ :
            show images
        """
        if self.args.assemble_method == "tiff":
            # assemble_func = self.__assemble_microscopy_volumne
            # assemble_func = self._assemble_microscopy_volume_memmap
            assemble_func = self.assemble_microscopy_volumne
        elif self.args.assemble_method == "zarr":
            assemble_func = self.assemble_microscopy_volume_zarr
        else:
            raise KeyError(f"Only support method tiff or zarr, but got {self.args.assemble_method}")
        # Do assemble
        assemble_func(zrange, xrange, yrange, source, output_path)

    def assemble_microscopy_volumne(self, zrange, xrange, yrange, source, output_path):
        C0, C1, C2 = self.kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']

        for c in range(self.kwargs['output_channel']):
            os.makedirs(output_path + '_' + str(c), exist_ok=True)

        last_block = None
        current_x_position = 0
        for nx in tqdm(range(len(xrange))):
            one_column = []
            for nz in range(len(zrange)):
                one_row = []
                for ny in range(len(yrange)):
                    # get weight
                    if nx == len(xrange) - 1:
                        nx = -1
                    if ny == len(yrange) - 1:
                        ny = -1
                    if nz == len(zrange) - 1:
                        nz = -1

                    iz = zrange[nz]
                    ix = xrange[nx]
                    iy = yrange[ny]

                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny, size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)

                    # load and crop
                    tini = time.time()
                    x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')
                    cropped = x[:, C0:-C0, C1:-C1, C2:-C2]

                    w = np.stack([w] * cropped.shape[0], axis=0)
                    cropped = np.multiply(cropped, w)
                    if len(one_row) > 0:
                        one_row[-1][:, :, :, -S2:] = one_row[-1][:, :, :, -S2:] + cropped[:, :, :, :S2]
                        one_row.append(cropped[:, :, :, S2:])
                    else:
                        one_row.append(cropped)

                #print("one row time : ", time.time() - tini)

                one_row = np.concatenate(one_row, axis=3)  # (C, Z, X, Y)
                one_row = np.transpose(one_row, (0, 2, 1, 3))  # (C, X, Z, Y)

                if len(one_column) > 0:
                    one_column[-1][:, :, -S0:, :] = one_column[-1][:, :, -S0:, :] + one_row[:, :, :S0, :]
                    one_column.append(one_row[:, :, S0:, :])
                else:
                    one_column.append(one_row)

            one_column = np.concatenate(one_column, axis=2).astype(np.float32)  # (C, X, Z, Y)

            if last_block is not None:
                one_column[:, :S1, ::] = one_column[:, :S1, ::] + last_block

            for xx in range(0, one_column.shape[1] - S1):
                for c in range(one_column.shape[0]):
                    slice_data = one_column[c, xx, ::]
                    # Clip to valid range for integer dtypes
                    if self.save_image_datatype == 'uint8':
                        slice_data = np.clip(slice_data, 0, 255)
                    elif self.save_image_datatype == 'uint16':
                        slice_data = np.clip(slice_data, 0, 65535)
                    tiff.imwrite(os.path.join(output_path + '_' + str(c), 'slice_x_' + str(current_x_position + xx).zfill(4) + '.tif'),
                                 slice_data.astype(np.dtype(self.save_image_datatype)))

            last_block = one_column[:, -S1:, :, :].copy()
            current_x_position += one_column.shape[1] - S1

    def assemble_microscopy_volume_zarr(self, zrange, xrange, yrange, source, output_path):
        # Compute final shape for zarr store
        def compute_final_shape(ori_img_path, Cz, Cx, Cy, Sz, Sx, Sy, dz, dy, dx, zstep, ystep, xstep, weight_shape):
            with tiff.TiffFile(ori_img_path) as t:
                ori_z, ori_y, ori_x = t.series[0].shape
            # Step 1: First, pad by crop margin (C) on both sides to preserve original size after assembly
            Dz, Dy, Dx = ori_z + 2*Cz, ori_y + 2*Cy, ori_x + 2*Cx
            # Step 2: Then, pad to make divisible by patch shape
            Nz, Ny, Nx = np.ceil(Dz / dz) * dz, np.ceil(Dy / dy) * dy, np.ceil(Dx / dx) * dx

            patches_z, patches_y, patches_x = np.ceil(Nz / eval(zstep)), np.ceil(Ny / eval(ystep)), np.ceil(Nx / eval(xstep))
            final_shape = weight_shape[0] * patches_z - (patches_z - 1) * Sz, weight_shape[1] * patches_y - (patches_y - 1) * Sy, weight_shape[2] * patches_x - (patches_x - 1) * Sx - Sx
            print(f'Final zarr output shape is: {final_shape}')
            return final_shape

        C0, C1, C2 = self.kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']

        # Compute final output shape (Z, Y, X) and create zarr store
        fZ, fY, fX = compute_final_shape(
            os.path.join(self.kwargs['root_path'], self.kwargs['image_path'][0]),
            C0, C1, C2, S0, S1, S2,
            self.kwargs["upsample_params"]["size"][0],
            self.kwargs["upsample_params"]["size"][1],
            self.kwargs["upsample_params"]["size"][2],
            self.kwargs['assemble_params']['zrange'][-1],
            self.kwargs['assemble_params']['yrange'][-1],
            self.kwargs['assemble_params']['xrange'][-1],
            self.kwargs['assemble_params']['weight_shape'],
        )
        fZ, fY, fX = int(fZ), int(fY), int(fX)
        C = self.kwargs['output_channel']
        save_dtype = np.dtype(self.save_image_datatype)

        # TCZYX 5D shape for OME-Zarr compatibility
        # Store X as zarr Z-axis (side view), so TCZYX = (1, C, X, Z, Y)
        shape_5d = (1, C, fX, fZ, fY)
        chunks_5d = (1, 1, 20, 512, 512)

        zarr_path = output_path + '_' + str(C) + '.zarr'
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.open_group(store, mode="w")
        zarr_out = root.create_dataset(
            "0",
            shape=shape_5d,
            chunks=chunks_5d,
            dtype=save_dtype,
            fill_value=0,
        )
        print(f"[zarr] Created store: {zarr_path}, shape TCZYX={shape_5d}")

        last_block = None
        current_x_position = 0
        for nx in tqdm(range(len(xrange))):
            one_column = []
            for nz in range(len(zrange)):
                one_row = []
                for ny in range(len(yrange)):
                    # get weight
                    if nx == len(xrange) - 1:
                        nx = -1
                    if ny == len(yrange) - 1:
                        ny = -1
                    if nz == len(zrange) - 1:
                        nz = -1

                    iz = zrange[nz]
                    ix = xrange[nx]
                    iy = yrange[ny]

                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny, size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)
                    # load and crop
                    x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')
                    cropped = x[:, C0:-C0, C1:-C1, C2:-C2]

                    w = np.stack([w] * cropped.shape[0], axis=0)
                    cropped = np.multiply(cropped, w)
                    if len(one_row) > 0:
                        one_row[-1][:, :, :, -S2:] = one_row[-1][:, :, :, -S2:] + cropped[:, :, :, :S2]
                        one_row.append(cropped[:, :, :, S2:])
                    else:
                        one_row.append(cropped)

                one_row = np.concatenate(one_row, axis=3)  # (C, Z, X, Y)
                one_row = np.transpose(one_row, (0, 2, 1, 3))  # (C, X, Z, Y)

                if len(one_column) > 0:
                    one_column[-1][:, :, -S0:, :] = one_column[-1][:, :, -S0:, :] + one_row[:, :, :S0, :]
                    one_column.append(one_row[:, :, S0:, :])
                else:
                    one_column.append(one_row)

            one_column = np.concatenate(one_column, axis=2).astype(np.float32)  # (C, X, Z, Y)

            if last_block is not None:
                one_column[:, :S1, ::] = one_column[:, :S1, ::] + last_block

            # Write block to zarr: TCZYX = (1, C, X, Z, Y)
            # one_column is (C, X, Z, Y), directly maps to zarr (C, Z_zarr, Y_zarr, X_zarr)
            block = one_column[:, :one_column.shape[1] - S1, :, :]
            if self.save_image_datatype == 'uint8':
                block = np.clip(block, 0, 255)
            elif self.save_image_datatype == 'uint16':
                block = np.clip(block, 0, 65535)
            block = block.astype(save_dtype)
            x_len = block.shape[1]
            zarr_out[0, :, current_x_position:current_x_position + x_len, :, :] = block

            last_block = one_column[:, -S1:, :, :].copy()
            current_x_position += one_column.shape[1] - S1

        print(f"[zarr] Assembly complete: {zarr_path}")

    def _test_time_augementation(self, x, method):
        axis_mapping_func = {"Z": 0, "X": 2, "Y": 3}
        # x shape: (Z, C, X, Y)
        if method == None:
            return x
        elif method.startswith('flip'):
            x = torch.flip(x, dims=[axis_mapping_func[method[-1]]])
            return x
        elif method == 'transpose':
            x = x.permute(0, 1, 3, 2)
            return x


    def _precompute_weights(self, zrange, xrange, yrange):
        S0, S1, S2 = self.kwargs['assemble_params']['S']
        weights_cache = {}
        for iz in zrange:
            for ix in xrange:
                for iy in yrange:
                    nz = zrange.index(iz)
                    nx = xrange.index(ix)
                    ny = yrange.index(iy)
                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny,
                                              size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)
                    weights_cache[(iz, ix, iy)] = w
        return weights_cache

    def _do_upsample(self, x0):
        assert self.upsample is not None, "upsample is None call get_model first to update upsample"
        return self.upsample(x0)


if __name__ == "__main__":
    tester = MicroTest()

    if tester.kwargs['assemble_params']['testwhole']:
        x0 = tester.get_data()
        dz, dx, dy = tester.kwargs['assemble_params']['dx_shape']
        Cz, Cx, Cy = tester.kwargs['assemble_params']['C']
        _, _, Dz, Dx, Dy = x0[0].shape

        # Add C padding on both sides (same as test_cleaned.py) before calculating grid
        Dz_padded, Dx_padded, Dy_padded = Dz + 2*Cz, Dx + 2*Cx, Dy + 2*Cy
        print(f"Original size: {Dz}, {Dx}, {Dy}")
        print(f"After C padding: {Dz_padded}, {Dx_padded}, {Dy_padded}")

        Nz, Nx, Ny = ((Dz_padded // dz + 1) * dz, (Dx_padded // dx + 1) * dx, (Dy_padded // dy + 1) * dy)
        print(f"Final padded size (divisible by dx_shape): {Nz}, {Nx}, {Ny}")

        zstep, ystep, xstep = tester.kwargs['assemble_params']['zrange'][-1], tester.kwargs['assemble_params']['yrange'][
            -1], tester.kwargs['assemble_params']['xrange'][-1]
        (zstep, ystep, xstep) = (int(eval(str(x))) for x in (zstep, ystep, xstep))
        zrange = [0, Nz, zstep]
        xrange = [0, Nx, xstep]
        yrange = [0, Ny, ystep]

    else:
        zrange = tester.kwargs['assemble_params']['zrange']
        yrange = tester.kwargs['assemble_params']['yrange']
        xrange = tester.kwargs['assemble_params']['xrange']


    zrange = range(*[eval(str(x)) for x in zrange])
    xrange = range(*[eval(str(x)) for x in xrange])
    yrange = range(*[eval(str(x)) for x in yrange])

    if tester.args.targets is not None:
        for target in tester.args.targets:
            tester.show_or_save_assemble_microscopy(zrange=zrange, xrange=xrange, yrange=yrange,
                                                    source=os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, target + '/'),
                                                    # output_path="tmp_xy.tif",
                                                    output_path=os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, target + '_assemble')#
                                                    )

    if tester.args.reslice:
        up = torch.nn.Upsample(scale_factor=(8, 1), mode='bilinear', align_corners=True)
        print("reslicing....")
        x0 = tester.get_data()
        print('Volume shape:  ', x0[0].shape)

        C = tester.kwargs['assemble_params']['C']

        for c in range(len(x0)):
            print('c', c)
            os.makedirs(os.path.join(tester.kwargs['root_path'], tester.kwargs['dataset'], 'ori_' + str(c) + '/'), exist_ok=True)
            x0[c] = (x0[c] - x0[c].min()) / (x0[c].max() - x0[c].min())
            # cropping
            x0[c] = x0[c][:, :, C[0] // 8:-C[0] // 8, C[1]:-C[1], C[2]:-C[2]]
            print('Volume shape:  ', x0[c].shape)
            for x in range(x0[c].shape[3]):
                slice = up(x0[c][:, :, :, x, :])
                slice = slice[0, 0, :1264, :2720]

                tiff.imwrite(os.path.join(tester.kwargs['root_path'], tester.kwargs['dataset'], 'ori_' + str(c) + '/',
                            f'slice_{x}.tif'), (slice.numpy() * 255).astype(np.uint8))


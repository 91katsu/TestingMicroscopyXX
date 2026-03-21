import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt
import json
import argparse
import os, importlib, sys

import torch.nn as nn
import time
import tifffile as tiff


def load_pth(gan, root, epoch, model_names):
    for name in model_names:
        setattr(gan, name, torch.load(os.path.join(root, f"{name}_model_epoch_{str(epoch)}.pth"),
                                      map_location=torch.device('cpu')))
    return gan


def import_model(root, model_name):
    model_path = os.path.join(root, f"{model_name}.py")
    module_name = f"dynamic_model_{model_name}"

    # Create the spec
    spec = importlib.util.spec_from_file_location(module_name, model_path)

    # Create the module
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module
    spec.loader.exec_module(module)

    return module


def read_json_to_args(json_file):
    with open(json_file, 'r') as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    return args


class ModelProcesser:
    def __init__(self, kwargs, model, gpu=True, augmentation='decode', fp16=False, device='cuda:0'):
        """
          A class that handles processing and augmentation of super-resolution model,
          Attributes:
              args (Namespace): Command-line or configuration arguments.
              kwargs (dict): Additional configuration parameters.
              model: Neural network model instance (either AE or GAN).
              upsample_size (tuple): Optional dimensions for upsampling.
              gpu (bool): Whether GPU processing should be used.
          """
        self.kwargs = kwargs
        self.model = model
        self.gpu = gpu
        self.augmentation = augmentation
        self.fp16 = fp16
        self.device = torch.device(device)

    def get_model_result(self, x0, input_augmentation, ii=None):
        """
           Process input tensor using the appropriate model type.

           Args:
               x0 (Tensor): Input tensor to process.
               input_augmentation: Augmentation method to apply.

           Returns:
               tuple: (XupX, Xup) - Processed output tensor and upsampled input tensor.
        """
        if self.kwargs['model_type'] == 'AE':
            XupX, Xup, outall, hbranch = self.get_ae_out(x0, input_augmentation, ii=ii)
        elif self.kwargs['model_type'] == 'GAN':
            raise NotImplementedError("GAN model processing not implemented.")
        elif self.kwargs['model_type'] == 'VQQ2':
            XupX, Xup, outall, hbranch = self.get_vqq_out(x0, input_augmentation, ii=ii)
        return XupX, Xup, outall, hbranch

    def get_ae_out(self, x0, method, ii=None):
        """
        Process input tensor through encoder then decoder with augmentation.

        Args:
            x0 (Tensor): Input tensor.
            method (list): List of augmentation methods to apply.

        Returns:
            tuple: (XupX, Xup) - Averaged output from multiple augmentations and upsampled input.
        """
        if self.augmentation == "decode":
            # decode only augmentation
            _, _, hbranch = self.get_ae_encode(x0)

            if ii is not None:
                iz, ix, iy = ii
                #hb2 = np.load('/media/ghc/Ghc_data3/BRC/aisr/aisr122424/enhanced/hbranchA/hbranch_' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.npy')
                #hb2 = torch.from_numpy(hb2).float().cuda()
                #hbranch = (hbranch + hb2) / 2.0  # average with the precomputed hbranch
                # max pooling
                #hbranch = torch.stack([hbranch, hb2], dim=-1).max(dim=-1)[0]  # (Z, C, X, Y)

            out_aug = []
            for i, aug in enumerate(method):  # (Z, C, X, Y)
                XupX, _ = self.get_ae_decode(hbranch, aug, ii)
                out_aug.append(XupX)
        else:
            # both encode and decode augmentation
            out_aug = []
            for i, aug in enumerate(method):  # (Z, C, X, Y)
                _, _, hbranch = self.get_ae_encode(x0, aug)
                XupX, _ = self.get_ae_decode(hbranch, aug)
                out_aug.append(XupX)

        out_aug = torch.stack(out_aug, -1).cpu()
        XupX = torch.mean(out_aug, -1)

        # upsample the original to Xup for display
        target_shape = XupX.permute(1, 2, 3, 0).unsqueeze(0).shape[2:]
        Xup = torch.nn.Upsample(size=target_shape, mode='trilinear')(
            x0.float().permute(1, 2, 3, 0).unsqueeze(0))  # (1, C, X, Y, Z)
        Xup = Xup[0, :, ::].permute(3, 0, 1, 2).detach().to('cpu')  # .numpy()  # (Z, C, X, Y))

        #print('XupX shape:', XupX.shape)
        #print('Xup shape:', Xup.shape)

        return XupX, Xup, out_aug, hbranch

    def get_ae_encode(self, x0, method=None):
        """
           Process input through encoder part of autoencoder.

           Args:
               x0 (Tensor): Input tensor.
               method (str, optional): Augmentation method for encoding phase.

           Returns:
               tuple: (Xup, _, hbranch) - Upsampled input, unused value, and latent representation.
        """
        if self.gpu:
            x0 = x0.to(self.device, non_blocking=True)
        if self.augmentation == "encode":
            x0 = self._test_time_augementation(x0, method=method)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                _, posterior, hbranch = self.model.forward(x0, sample_posterior=False)
            if self.kwargs['hbranchz']:
                hbranch = posterior.sample()

        # extra downsample to reduce the size of hbranch for lower expansion rate (<8)
        if self.kwargs['downbranch'] > 1:
            hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
            hbranch = nn.MaxPool3d((1, 1, self.kwargs['downbranch']))(hbranch)  # extra downsample, (1, C, X, Y, Z/2)
            hbranch = hbranch.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)

        return _, _, hbranch

    def get_ae_decode(self, hbranch, method, ii=None):
        """
           Process latent representation through decoder part of autoencoder.

           Args:
               hbranch (Tensor): Latent representation to decode.
               method (str): Augmentation method for decoding phase.

           Returns:
               Tensor: XupX - Decoded and potentially augmented output.
        """
        if self.gpu:
            hbranch = hbranch.to(self.device)
        if self.augmentation == "decode":
            hbranch = self._test_time_augementation(hbranch, method=method)
        if self.kwargs['hbranchz']:
            hbranch = self.model.decoder.conv_in(hbranch)

        hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (C, X, Y, Z)

        if ii is not None:
            iz, ix, iy = ii
            #hb2 = np.load('/media/ghc/Ghc_data3/BRC/aisr/aisr122424/enhanced/hbranch/hbranch_' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.npy')
            #hb2 = torch.from_numpy(hb2).float().cuda()
            #hbranch = (hbranch + hb2) / 2.0  # average with the precomputed hbranch
            #max pooling
            #print(hbranch.shape, hb2.shape)
            #hbranch = torch.stack([hbranch, hb2], dim=-1).max(dim=-1)[0]  # (Z, C, X, Y)

        torch.cuda.synchronize(self.device)
        out = self.model.net_g(hbranch, method='decode')
        Xout = out['out0'].detach()#.to('cpu')  # (1, C, X, Y, Z) # , non_blocking=True, non_blocking=True
        torch.cuda.synchronize(self.device)

        # (1, C, X, Y, Z)
        XupX = Xout[0, :].permute(3, 0, 1, 2)  # (Z, C, X, Y)
        XupX = self._test_time_augementation(XupX, method=method)
        return XupX, hbranch

    def _test_time_augementation(self, x, method):
        """
           Apply test-time augmentation to input tensor.

           Args:
               x (Tensor): Tensor to augment.
               method (str): String specifying augmentation method.

           Returns:
               Tensor: Augmented tensor.
        """
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

    def get_vqq_out(self, x0, method, ii=None):
        """
        Process input tensor through VQQ model with quantization.

        Args:
            x0 (Tensor): Input tensor.
            method (list): List of augmentation methods to apply.
            ii (tuple): Optional indices for position information.

        Returns:
            tuple: (XupX, Xup, out_aug, hbranch) - Averaged output from multiple augmentations and upsampled input.
        """
        import torch.nn.functional as F
        
        if self.gpu:
            x0 = x0.to(self.device, non_blocking=True)

        out_aug = []
        
        for i, aug in enumerate(method):  # (Z, C, X, Y)
            # Apply input augmentation if needed
            x_input = self._test_time_augementation(x0, method=aug) if aug else x0
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    # Encode
                    h, hbranch, hz = self.model.encoder(x_input)  # (32, 4, 32, 32)
                    h = self.model.quant_conv(h)
                    quant, emb_loss, info = self.model.quantize(h)
                    
                    # Optional codebook checking (from test_vqq.py)
                    checking_codebook = self.kwargs.get('checking_codebook', False)
                    if checking_codebook:
                        B, C, H, W = quant.shape
                        # Get indices robustly
                        idx = info["min_encoding_indices"] if isinstance(info, dict) else info[-1]
                        # Make sure dtype/device are right and reshape to (B, H, W)
                        idx = idx.view(B, H, W).to(dtype=torch.long, device=self.model.quantize.embedding.weight.device)
                        # Dequantize: (*, C) then permute to (B, C, H, W)
                        quant2 = F.embedding(idx, self.model.quantize.embedding.weight)  # (B, H, W, C)
                        quant2 = quant2.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
                        quant = quant2
                    
                    # Extra downsampling if needed
                    downbranch = self.kwargs.get('downbranch', 1)
                    h = quant
                    if downbranch > 1:
                        h = h.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
                        h = torch.nn.MaxPool3d((1, 1, downbranch))(h)  # extra downsample
                        h = h.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)
                    
                    # Decode
                    h = self.model.decoder.conv_in(h)  # (16, 256, 16, 16)
                    h = h.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
                    
                    # Generate output
                    XupX = self.model.net_g(h[:, :, :, :, :], method='decode')['out0'].detach().cpu().float()
                    # Remove batch dimension and rearrange: (1, C, X, Y, Z) -> (Z, C, X, Y)
                    XupX = XupX[0, :].permute(3, 0, 1, 2)  # (Z, C, X, Y)

                    XupX = self._test_time_augementation(XupX, method=aug)

                    # Apply output augmentation if needed
                    decode_augmentation = self.kwargs.get('decode_augmentation', False)
                    if decode_augmentation:
                        # Additional augmentations from test_vqq.py
                        aug_outputs = [XupX]
                        #aug_outputs.append(self.model.net_g(h[:, :, :, :, :].permute(0, 1, 3, 2, 4), method='decode')['out0'].permute(0, 1, 3, 2, 4).squeeze().permute(0, 2, 1).detach().cpu().float())
                        #aug_outputs.append(torch.flip(self.model.net_g(torch.flip(h, dims=[2]), method='decode')['out0'], dims=[2]).squeeze().permute(0, 2, 1).detach().cpu().float())
                        #aug_outputs.append(torch.flip(self.model.net_g(torch.flip(h, dims=[3]), method='decode')['out0'], dims=[3]).squeeze().permute(0, 2, 1).detach().cpu().float())
                        XupX = torch.stack(aug_outputs, dim=-1).mean(dim=-1)
            
            out_aug.append(XupX)
        
        out_aug = torch.stack(out_aug, -1)
        XupX = torch.mean(out_aug, -1)

        # upsample the original to Xup for display
        target_shape = XupX.permute(1, 2, 3, 0).unsqueeze(0).shape[2:]
        Xup = torch.nn.Upsample(size=target_shape, mode='trilinear')(
            x0.float().permute(1, 2, 3, 0).unsqueeze(0))  # (1, C, X, Y, Z)
        Xup = Xup[0, :, ::].permute(3, 0, 1, 2).detach().to('cpu')  # .numpy()  # (Z, C, X, Y))
        #print('XupX shape:', XupX.shape)
        #print('Xup shape:', Xup.shape)
        return XupX, Xup, out_aug, hbranch


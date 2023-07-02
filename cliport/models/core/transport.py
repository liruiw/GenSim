import numpy as np
import cliport.models as models
from cliport.utils import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transport(nn.Module):

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        """Transport (a.k.a Place) module."""
        super().__init__()

        self.iters = 0
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size
  
        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        # Crop before network (default from Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        model = models.names[stream_one_fcn]
        self.key_resnet = model(self.in_shape, self.output_dim, self.cfg, self.device)
        self.query_resnet = model(self.kernel_shape, self.kernel_dim, self.cfg, self.device)
        print(f"Transport FCN: {stream_one_fcn}")

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        output_shape = output.shape

        # a hack around the batch size 1. The shape needs to tile back.
        channel_num = in1.shape[0] // in0.shape[0]
        output = torch.stack([output[i,i*channel_num:(i+1)*channel_num] for i in range(len(output))], dim=0)
        if softmax:
            output = output.reshape((len(output), -1))
            output = F.softmax(output, dim=-1)
        output = output.reshape(len(output),channel_num,output_shape[2],output_shape[3])

        return output

    def transport(self, in_tensor, crop):
        logits = self.key_resnet(in_tensor)
        kernel = self.query_resnet(crop)
        return logits, kernel

    def forward(self, inp_img, p, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape =  input_data.shape
        if len(inp_shape) == 3:
            inp_shape = (1,) + inp_shape
        input_data = input_data.reshape(in_shape) # [B W H D]
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = p  + self.pad_size # np.array([p[0], p[1]])

        # Crop before network (default from Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2) # [B D W H]

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        logits, kernel = self.transport(in_tensor, crop)

        # TODO(Mohit): Crop after network. Broken for now.
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)

        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        # kernel = crop[:, :, p[0]:(p[0] + self.crop_size), p[1]:(p[1] + self.crop_size)]

        return self.correlate(logits, kernel, softmax)


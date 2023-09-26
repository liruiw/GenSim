import numpy as np
import torch
import torch.nn.functional as F

from cliport.models.core.attention import Attention
import cliport.models as models
import cliport.models.core.fusion as fusion


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]  # resnet_lat.REsNet45_10s
        stream_two_model = models.names[stream_two_fcn]  # clip_ligunet_lat.CLIP_LIGUnet_lat

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        if len(inp_img.shape) < 4:
            inp_img = inp_img[None]

        if type(inp_img) is not torch.Tensor:
            in_data = inp_img # .reshape(in_shape)
            in_tens = torch.from_numpy(in_data.copy()).to(dtype=torch.float, device=self.device)  # [B W H 6]
        else:
            in_data = inp_img
            in_tens = in_data

        # [B W H 6]
        in_tens = torch.nn.functional.pad(in_tens, tuple(self.padding[[2,1,0]].reshape(-1)), mode='constant')

        # Rotation pivot.
        pv = np.array(in_tens.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]

        # in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        # make n copies, but keep batchsize
        in_tens = [in_tens] * self.n_rotations
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = self.attend(torch.cat(in_tens, dim=0), lang_goal)
        
        # Rotate back output.
        logits = self.rotator([logits], reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img[0].shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]
        output_shape = logits.shape

        # logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(len(logits), -1)
        if softmax:
            output = F.softmax(output, dim=-1)
        return output.view(output_shape)


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x



class TwoStreamAttentionLangFusionLatReduce(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)
        
        del self.attn_stream_one
        del self.attn_stream_two
        
        stream_one_fcn = 'plain_resnet_reduce_lat'
        stream_one_model = models.names[stream_one_fcn]
        stream_two_fcn = 'clip_ling'
        stream_two_model = models.names[stream_two_fcn]
        
        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x
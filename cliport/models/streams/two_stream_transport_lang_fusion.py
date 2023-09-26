import torch
import numpy as np

import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.core.transport import Transport


class TwoStreamTransportLangFusion(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport2(self, in_tensor, crop, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        if len(inp_img.shape) < 4:
            inp_img = inp_img[None]

        if type(inp_img) is not torch.Tensor:
            in_data = inp_img # .reshape(in_shape)
            in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]
        else:
            in_data = inp_img
            in_tens = in_data

        in_tensor = torch.nn.functional.pad(in_tens, tuple(self.padding[[2,1,0]].reshape(-1)), mode='constant')
        if type(p[0]) is not torch.Tensor:
            p = torch.FloatTensor(p)[None]

        in_tensors = []
        crops = []

        # this for loop is fast.
        for i in range(len(in_tensor)):
            in_tensor_i = in_tensor[[i]]           
            # Rotation pivot.
            pv = p[i] + self.pad_size

            # Crop before network (default for Transporters CoRL 2020).
            hcrop = self.pad_size
            in_tensor_i = in_tensor_i.permute(0, 3, 1, 2)

            crop = [in_tensor_i] * self.n_rotations
            crop = self.rotator(crop, pivot=pv.float())
            crop = torch.cat(crop, dim=0)
            crop = crop[:, :, int(pv[0]-hcrop):int(pv[0]+hcrop), int(pv[1]-hcrop):int(pv[1]+hcrop)]

            in_tensors.append(in_tensor_i)
            crops.append(crop)

        logits, kernels = self.transport(torch.cat(in_tensors,dim=0), torch.cat(crops, dim=0), lang_goal) #crops.shape:(8, 36, 6, 64, 64)
        res =  self.correlate(logits, kernels, softmax)
        return res
        
class TwoStreamTransportLangFusionLat(TwoStreamTransportLangFusion):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel
    
    
class TwoStreamTransportLangFusionLatReduce(TwoStreamTransportLangFusionLat):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        
        del self.query_stream_one
        del self.query_stream_two
        # del self.key_stream_one
        # del self.key_stream_two

        stream_one_fcn = 'plain_resnet_reduce_lat'
        stream_one_model = models.names[stream_one_fcn]
        stream_two_fcn = 'clip_ling'
        stream_two_model = models.names[stream_two_fcn]
        
        
        
        # self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        # self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel





class TwoStreamTransportLangFusionLatReduceOneStream(TwoStreamTransportLangFusionLatReduce):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        
        del self.query_stream_one
        del self.query_stream_two

        

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.key_stream_one(crop)
        query_out_two = self.key_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel




class TwoStreamTransportLangFusionLatPretrained18(TwoStreamTransportLangFusionLat):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)
        
        del self.query_stream_one
        del self.query_stream_two
        # del self.key_stream_one
        # del self.key_stream_two
        stream_one_fcn = 'pretrained_resnet18'
        stream_one_model = models.names[stream_one_fcn]
        stream_two_fcn = 'clip_ling'
        stream_two_model = models.names[stream_two_fcn]
        
        # self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        # self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel
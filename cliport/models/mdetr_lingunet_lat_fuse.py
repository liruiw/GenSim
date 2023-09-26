import torch
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor, nn
import copy
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up

from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat
from cliport.models.backbone_full import Backbone
from cliport.models.misc import NestedTensor
from cliport.models.position_encoding import build_position_encoding
from transformers import RobertaModel, RobertaTokenizerFast



class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class MDETRLingUNetLat_fuse(nn.Module):
    """ CLIP RN50 with U-Net skip connections and lateral connections """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(MDETRLingUNetLat_fuse, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of mdetr
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        self.lang_fusion_type = self.cfg['train']['lang_fusion_type']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.preprocess = preprocess

        self.backbone = Backbone('resnet101', True, True, False)
        self.position_embedding = build_position_encoding()
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)

        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=256,
            dropout=0.1,
        )
        encoder_layer = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, 6, None)
        mdter_checkpoint = torch.load('/home/yzc/shared/project/GPT-CLIPort/ckpts/mdetr_pretrained_resnet101_checkpoint.pth', map_location="cpu")['model']

        checkpoint_new = {}
        for param in mdter_checkpoint:
            if 'transformer.text_encoder' in param or 'transformer.encoder.' in param or 'input_proj' in param or 'resizer' in param:
                param_new = param.replace('transformer.','')
                checkpoint_new[param_new] = mdter_checkpoint[param]
            elif 'backbone.0.body' in param:
                param_new = param.replace('backbone.0.body', 'backbone.body')
                checkpoint_new[param_new] = mdter_checkpoint[param]

        self.load_state_dict(checkpoint_new, True)
        self._build_decoder()


    def _build_decoder(self):
        # language
        self.up_fuse1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_fuse2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up_fuse3 = nn.UpsamplingBilinear2d(scale_factor=8)

        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 768
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim+256, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up1 = Up(2048+256, 1024 // self.up_factor, self.bilinear)
        self.lat_fusion1 = FusionConvLat(input_dim=1024+512, output_dim=512)

        self.up2 = Up(1024+256, 512 // self.up_factor, self.bilinear)
        self.lat_fusion2 = FusionConvLat(input_dim=512+256, output_dim=256)

        self.up3 = Up(512+256, 256 // self.up_factor, self.bilinear)
        self.lat_fusion3 = FusionConvLat(input_dim=256+128, output_dim=128)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion4 = FusionConvLat(input_dim=128+64, output_dim=64)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion5 = FusionConvLat(input_dim=64+32, output_dim=32)

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion6 = FusionConvLat(input_dim=32+16, output_dim=16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def encode_image(self, img):
        img = NestedTensor.from_tensor_list(img)
        with torch.no_grad():
            xs = self.backbone(img)
            out = []
            pos = []
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self.position_embedding(x).to(x.tensors.dtype))
            return out, pos


    def encode_text(self, x):
        with torch.no_grad():
            tokenized = self.tokenizer.batch_encode_plus(x, padding="longest", return_tensors="pt").to(self.device)
            encoded_text = self.text_encoder(**tokenized)

            # Transpose memory because pytorch's attention expects sequence first
            text_memory = encoded_text.last_hidden_state.transpose(0, 1)
            text_memory_mean = torch.mean(text_memory, 0)
            # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # Resize the encoder hidden states to be of the same d_model as the decoder
            text_memory_resized = self.resizer(text_memory)
        return text_memory_resized, text_attention_mask, text_memory_mean

    def forward(self, x, lat, l):

        x = self.preprocess(x, dist='mdetr')

        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB

        x = x.permute(0, 1, 3, 2)


        with torch.no_grad():
            features, pos = self.encode_image(x)
            x1, mask = features[-1].decompose()
            x2, _ = features[-2].decompose()
            x3, _ = features[-3].decompose()
            x4, _ = features[-4].decompose()
            #print(x1.shape, x2.shape, x3.shape, x4.shape)
            src = self.input_proj(x1)
            pos_embed = pos[-1]
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            device = self.device
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
            if x.shape[0] == 1 or x.shape[0] == 36:
                l = [l]
                text_memory_resized, text_attention_mask, l_input = self.encode_text(l)
            else:
                text_memory_resized, text_attention_mask, l_input = self.encode_text(l)
            # l_input = l_input.view(1, -1)
            # text_memory_resized = text_memory_resized.repeat(1, src.shape[1], 1)
            # text_attention_mask = text_attention_mask.repeat(src.shape[1], 1)
            #print(src.shape, text_memory_resized.shape, mask.shape, text_attention_mask.shape)
            if (x.shape[0] > 8) and ((x.shape[0] % 36) == 0):
                text_memory_resized = text_memory_resized.repeat_interleave(36, dim=1)
                l_input = l_input.repeat_interleave(36, dim=0)
                text_attention_mask = text_attention_mask.repeat_interleave(36, dim=0)
            src = torch.cat([src, text_memory_resized], dim=0)
            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)
            img_memory, img_memory_all  = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        dim = img_memory.shape[-1]
        fuse1 = img_memory_all[-1][:h*w].permute(1,2,0).reshape(bs, dim, h, w)
        fuse2 = self.up_fuse1(img_memory_all[-2][:h*w].permute(1,2,0).reshape(bs, dim, h, w))
        fuse3 = self.up_fuse2(img_memory_all[-3][:h*w].permute(1,2,0).reshape(bs, dim, h, w))
        fuse4 = self.up_fuse3(img_memory_all[-4][:h*w].permute(1,2,0).reshape(bs, dim, h, w))

        assert x1.shape[1] == self.input_dim

        x1 = torch.cat((x1, fuse1), 1)
        x2 = torch.cat((x2, fuse2), 1)
        x3 = torch.cat((x3, fuse3), 1)
        x4 = torch.cat((x4, fuse4), 1)

        x = self.conv1(x1)
        x = self.lang_fuser1(x, l_input, x2_mask=None, x2_proj=self.lang_proj1)
        x = self.up1(x, x2)
        x = self.lat_fusion1(x, lat[-6].permute(0, 1, 3, 2))

        x = self.lang_fuser2(x, l_input, x2_mask=None, x2_proj=self.lang_proj2)

        x = self.up2(x, x3)
        x = self.lat_fusion2(x, lat[-5].permute(0, 1, 3, 2))

        x = self.lang_fuser3(x, l_input, x2_mask=None, x2_proj=self.lang_proj3)
        x = self.up3(x, x4)
        x = self.lat_fusion3(x, lat[-4].permute(0, 1, 3, 2))
        x = self.layer1(x)
        x = self.lat_fusion4(x, lat[-3].permute(0, 1, 3, 2))

        x = self.layer2(x)
        x = self.lat_fusion5(x, lat[-2].permute(0, 1, 3, 2))

        x = self.layer3(x)
        x = self.lat_fusion6(x, lat[-1].permute(0, 1, 3, 2))

        x = self.conv2(x)

        x = F.interpolate(x, size=(in_shape[-1], in_shape[-2]), mode='bilinear')
        x = x.permute(0, 1, 3, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src
        output_all = []
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
            output_all.append(output)
        if self.norm is not None:
            output = self.norm(output)

        return output, output_all

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        print(self.normalize_before)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


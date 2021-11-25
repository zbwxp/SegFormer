import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..utils import PositionEmbeddingSine
from ..utils.deformable_transformer import *

class Deformable_encoder(nn.ModuleList):
    def __init__(self,
                 out_channel=256,
                 channels=[128, 320, 512],
                 conv_cfg=None,
                 norm_cfg={'type': 'BN', 'requires_grad': True},
                 act_cfg={'type': 'ReLU'}
                 ):
        super(Deformable_encoder, self).__init__()

        lateral_convs = []
        for channel in channels:
            lateral_convs.append(
                ConvModule(
                    channel,
                    out_channel,
                    1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        lateral_convs.append(
            ConvModule(
                channel,
                out_channel,
                1,
                padding=0,
                stride=2,
                conv_cfg=None,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

        self.add_module("res{}".format(3), lateral_convs[0])
        self.add_module("res{}".format(4), lateral_convs[1])
        self.add_module("res{}".format(5), lateral_convs[2])
        self.add_module("res{}".format(6), lateral_convs[3])
        self.lateral_convs = lateral_convs
        N_steps = out_channel // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        d_model = 256
        dim_feedforward = 1024
        dropout = 0.1
        activation = "relu"
        num_feature_levels = 4
        nhead = 8
        enc_n_points = 4
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        num_encoder_layers = 6
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.mask_features = None

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            matches = ["adapter", "layer", "mask_features"]
            if p.dim() > 1 and not any(x in name for x in matches):
                nn.init.xavier_uniform_(p)
        for name, m in self.named_modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        # xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        # constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def forward(self, features):
        features = features[1:]
        srcs = []
        pos = []
        mask = []
        for idx, feat in enumerate(features):
            lateral_conv = self.lateral_convs[idx]
            x = lateral_conv(feat)
            pos.append(self.pe_layer(x))
            mask.append(torch.zeros((x.size(0), x.size(2), x.size(3)),
                                    device=x.device, dtype=torch.bool))
            srcs.append(x)

        srcs.append(self.lateral_convs[-1](features[-1]))
        pos.append(self.pe_layer(srcs[-1]))
        mask.append(torch.zeros((srcs[-1].size(0), srcs[-1].size(2), srcs[-1].size(3)),
                                device=srcs[-1].device, dtype=torch.bool))

        encoder_results = \
            self.deformable_transformer_encoder(srcs, mask, pos)
        memory = encoder_results["memory"]
        encoded_maps = self.flat2feature(memory, encoder_results["spatial_shapes"],
                                         encoder_results["level_start_index"])
        # out = encoded_maps[-2]
        out = []
        for map in encoded_maps:
            out.append(F.interpolate(map, size=encoded_maps[-2].size()[-2:], mode="nearest"))

        out = torch.cat(out, dim=1)

        return out

    def deformable_transformer_encoder(self, srcs, masks, pos_embeds, query_embed=None):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        pos_embeds_flatten = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            pos_embeds_flatten.append(pos_embed)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        pos_embeds_flatten = torch.cat(pos_embeds_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.encoder.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape

        # query_embed, tgt = torch.split(query_embed, c, dim=1)
        # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        # reference_points = self.reference_points(query_embed).sigmoid()
        # init_reference_out = reference_points

        return {
            # "tgt": tgt,
            # "reference_points": reference_points,
            "memory": memory,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            # "query_embed": query_embed,
            "mask_flatten": mask_flatten,
            "pos": pos_embeds_flatten
        }

    def flat2feature(self, t, spatial_shapes, level_start_index):
        b, num, ch = t.size()
        feature_maps = []
        for idx, shape in zip(level_start_index, spatial_shapes):
            range = shape[0] * shape[1]
            feature_map = t[:, idx:idx + range, :].reshape(b, shape[0], shape[1], ch)
            feature_map = feature_map.permute(0, -1, 1, 2)
            feature_maps.append(feature_map)

        return feature_maps
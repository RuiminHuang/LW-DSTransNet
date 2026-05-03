# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
# import ml_collections
from einops import rearrange
import numbers

import numpy as np

from thop import profile

from .Config import get_DSTransNet_Student_config



class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        patch_size = _pair(patchsize)
        img_size = _pair(img_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 16 * 16 = 256

        self.patch_embeddings = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=patch_size, stride=patch_size)# 8 3 256 256 -> 8 3 16 16
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels)) # 位置编码已被取消
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"]) # 0.1

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        return x

    # def forward(self, x):
    #     if x is None:
    #         return None
    #     x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
    #     x = x.flatten(2)
    #     x = x.transpose(-1, -2)  # (B, n_patches, hidden)
    #     embeddings = x + self.position_embeddings
    #     embeddings = self.dropout(embeddings)
    #     return embeddings


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else: # only use this
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor


    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out
    # def forward(self, x):
    #     B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
    #     h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    #     x = x.permute(0, 2, 1) # (B, n_patch, hidden) -> (B, hidden, n_patch)
    #     x = x.contiguous().view(B, hidden, h, w) # (B, hidden, h, w)
    #     if self.scale_factor[0] > 1:
    #         x = nn.Upsample(scale_factor=self.scale_factor)(x)
    #     out = self.conv(x)
    #     out = self.norm(out)
    #     out = self.activation(out)
    #     return out




# spatial-embedded Single-head Channel-cross Attention (SSCA)
# SCTransNet
class Attention_org(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = 1
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        self.q2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        self.q3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        self.q4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
        self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)


        # ****************** useless ***************************************
        self.q1_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q1_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q2_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q2_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q3_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q3_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.q4_attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.q4_attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        b, c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))
        q2 = self.q2(self.mhead2(emb2))
        q3 = self.q3(self.mhead3(emb3))
        q4 = self.q4(self.mhead4(emb4))
        k = self.k(self.mheadk(emb_all))
        v = self.v(self.mheadv(emb_all))
        # k, v = kv.chunk(2, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q3 = rearrange(q3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q4 = rearrange(q4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        q3 = torch.nn.functional.normalize(q3, dim=-1)
        q4 = torch.nn.functional.normalize(q4, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape
        _, _, c2, _ = q2.shape
        _, _, c3, _ = q3.shape
        _, _, c4, _ = q4.shape
        _, _, c, _ = k.shape

        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn2 = (q2 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn3 = (q3 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn4 = (q4 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        attention_probs4 = self.softmax(self.psi(attn4))

        out1 = (attention_probs1 @ v)
        out2 = (attention_probs2 @ v)
        out3 = (attention_probs3 @ v)
        out4 = (attention_probs4 @ v)

        out_1 = out1.mean(dim=1)
        out_2 = out2.mean(dim=1)
        out_3 = out3.mean(dim=1)
        out_4 = out4.mean(dim=1)

        out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)
        out_2 = rearrange(out_2, 'b  c (h w) -> b c h w', h=h, w=w)
        out_3 = rearrange(out_3, 'b  c (h w) -> b c h w', h=h, w=w)
        out_4 = rearrange(out_4, 'b  c (h w) -> b c h w', h=h, w=w)

        O1 = self.project_out1(out_1)
        O2 = self.project_out2(out_2)
        O3 = self.project_out3(out_3)
        O4 = self.project_out4(out_4)
        weights = None

        return O1, O2, O3, O4, weights





# DSTransNet
# class Attention_org(nn.Module):
#     def __init__(self, config, vis, channel_num):
#         super(Attention_org, self).__init__()
#         self.vis = vis
#         self.KV_size = config.KV_size # channel_num[0] + channel_num[1] + channel_num[2] + channel_num[3]
#         self.channel_num = channel_num # channel_num[0], channel_num[1], channel_num[2], channel_num[3]
#         self.num_attention_heads = 1

#         self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
#         self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
#         self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
#         self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)

#         self.mheadq_C = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
#         self.mheadk_C = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
#         self.mheadv_C = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

#         self.q1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, 
#                             channel_num[0] * self.num_attention_heads, 
#                             kernel_size=3, stride=1, padding=1,
#                             groups=channel_num[0] * self.num_attention_heads // 2,
#                             bias=False)
#         self.q2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, 
#                             channel_num[1] * self.num_attention_heads, 
#                             kernel_size=3, stride=1, padding=1,
#                             groups=channel_num[1] * self.num_attention_heads // 2,
#                             bias=False)
#         self.q3 = nn.Conv2d(channel_num[2] * self.num_attention_heads,
#                             channel_num[2] * self.num_attention_heads,
#                             kernel_size=3, stride=1, padding=1,
#                             groups=channel_num[2] * self.num_attention_heads // 2,
#                             bias=False)
#         self.q4 = nn.Conv2d(channel_num[3] * self.num_attention_heads,
#                             channel_num[3] * self.num_attention_heads,
#                             kernel_size=3, stride=1, padding=1,
#                             groups=channel_num[3] * self.num_attention_heads // 2,
#                             bias=False)
#         self.q_C = nn.Conv2d(self.KV_size * self.num_attention_heads,
#                            self.KV_size * self.num_attention_heads,
#                            kernel_size=3, stride=1, padding=1,
#                            groups=self.KV_size * self.num_attention_heads,
#                            bias=False)
#         self.k_C = nn.Conv2d(self.KV_size * self.num_attention_heads,
#                            self.KV_size * self.num_attention_heads,
#                            kernel_size=3, stride=1, padding=1,
#                            groups=self.KV_size * self.num_attention_heads,
#                            bias=False)
#         self.v_C = nn.Conv2d(self.KV_size * self.num_attention_heads,
#                            self.KV_size * self.num_attention_heads,
#                            kernel_size=3, stride=1, padding=1,
#                            groups=self.KV_size * self.num_attention_heads,
#                            bias=False)
        
#         self.CFA_psi = nn.InstanceNorm2d(self.num_attention_heads)
#         self.CFA_softmax = Softmax(dim=3)

#         self.SSA_psi = nn.InstanceNorm2d(self.num_attention_heads)
#         self.SSA_softmax = Softmax(dim=3)


#         self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
#         self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
#         self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
#         self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)

#     def forward(self, emb1, emb2, emb3, emb4, emb_all):

#         # Step1 CFA Module
#         Q_C = self.q_C(self.mheadq_C(emb_all))# [1 480 16 16]
#         K_C = self.k_C(self.mheadk_C(emb_all))# [1 480 16 16]
#         V_C = self.v_C(self.mheadv_C(emb_all))# [1 480 16 16]
#         Q_C = rearrange(Q_C, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)# [1 1 480 256]
#         K_C = rearrange(K_C, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)# [1 1 480 256]
#         V_C = rearrange(V_C, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)# [1 1 480 256]


#         Q_C = torch.nn.functional.normalize(Q_C, dim=-1)# [1 1 480 256]
#         K_C = torch.nn.functional.normalize(K_C, dim=-1)# [1 1 480 256]
#         ch_similarity_matrix = (Q_C @ K_C.transpose(-1, -2)) / math.sqrt(self.KV_size)# [1 1 480 256] @ [1 1 256 480] = [1 1 480 480]
#         ch_similarity_matrix = self.CFA_softmax(self.CFA_psi(ch_similarity_matrix))# IN Feature 1, Softmax Dim 3
#         context_layer = ch_similarity_matrix @ V_C# [1 1 480 480] @ [1 1 480 256] = [1 1 480 256]


#         # Step2 SSA Module
#         q1 = self.q1(self.mhead1(emb1))# [1 32 16 16]
#         q2 = self.q2(self.mhead2(emb2))# [1 64 16 16]
#         q3 = self.q3(self.mhead3(emb3))# [1 128 16 16]
#         q4 = self.q4(self.mhead4(emb4))# [1 256 16 16]
#         q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)# [1 1 32 256]
#         q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)# [1 1 64 256]
#         q3 = rearrange(q3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)# [1 1 128 256]
#         q4 = rearrange(q4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)# [1 1 256 256]
        
        
#         q1 = torch.nn.functional.normalize(q1, dim=-1)# [1 1 32 256]
#         q2 = torch.nn.functional.normalize(q2, dim=-1)# [1 1 64 256]
#         q3 = torch.nn.functional.normalize(q3, dim=-1)# [1 1 128 256]
#         q4 = torch.nn.functional.normalize(q4, dim=-1)# [1 1 256 256]


#         attn1 = (q1 @ context_layer.transpose(-1, -2)) / math.sqrt(self.KV_size)# [1 1 32 256] @ [1 1 256 480] = [1 1 32 480]
#         attn2 = (q2 @ context_layer.transpose(-1, -2)) / math.sqrt(self.KV_size)# [1 1 64 256] @ [1 1 256 480] = [1 1 64 480]
#         attn3 = (q3 @ context_layer.transpose(-1, -2)) / math.sqrt(self.KV_size)# [1 1 128 256] @ [1 1 256 480] = [1 1 128 480]
#         attn4 = (q4 @ context_layer.transpose(-1, -2)) / math.sqrt(self.KV_size)# [1 1 256 256] @ [1 1 256 480] = [1 1 256 480]

#         attention_probs1 = self.SSA_softmax(self.SSA_psi(attn1))# IN Feature 1, Softmax Dim 3
#         attention_probs2 = self.SSA_softmax(self.SSA_psi(attn2))# IN Feature 1, Softmax Dim 3
#         attention_probs3 = self.SSA_softmax(self.SSA_psi(attn3))# IN Feature 1, Softmax Dim 3
#         attention_probs4 = self.SSA_softmax(self.SSA_psi(attn4))# IN Feature 1, Softmax Dim 3

#         out1 = (attention_probs1 @ context_layer)# [1 1 32 480] @ [1 1 480 256] = [1 1 32 256]
#         out2 = (attention_probs2 @ context_layer)# [1 1 64 480] @ [1 1 480 256] = [1 1 64 256]
#         out3 = (attention_probs3 @ context_layer)# [1 1 128 480] @ [1 1 480 256] = [1 1 128 256]
#         out4 = (attention_probs4 @ context_layer)# [1 1 256 480] @ [1 1 480 256] = [1 1 256 256]


#         # Step3 output process
#         out_1 = out1.mean(dim=1)# [1 32 256]
#         out_2 = out2.mean(dim=1)# [1 64 256]
#         out_3 = out3.mean(dim=1)# [1 128 256]
#         out_4 = out4.mean(dim=1)# [1 256 256]

#         b, c, h, w = emb1.shape
#         out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)# [1 32 16 16]
#         out_2 = rearrange(out_2, 'b  c (h w) -> b c h w', h=h, w=w)# [1 64 16 16]
#         out_3 = rearrange(out_3, 'b  c (h w) -> b c h w', h=h, w=w)# [1 128 16 16]
#         out_4 = rearrange(out_4, 'b  c (h w) -> b c h w', h=h, w=w)# [1 256 16 16]

#         O1 = self.project_out1(out_1)# [1 32 16 16]
#         O2 = self.project_out2(out_2)# [1 64 16 16]
#         O3 = self.project_out3(out_3)# [1 128 16 16]
#         O4 = self.project_out4(out_4)# [1 256 16 16]
#         weights = None

#         return O1, O2, O3, O4, weights



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

# Complementary Feed-forward Network (CFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def forward(self, x):
        x_3,x_5 = self.project_in(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x_3))
        x1_5 = self.relu5(self.dwconv5x5(x_5))
        x = torch.cat([x1_3, x1_5], dim=1)
        x = self.project_out(x)
        x = self.eca(x)
        return x


#  Spatial-channel Cross Transformer Block (SCTB)
class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')

        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)


    def forward(self, emb1, emb2, emb3, emb4):
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        emb_all = torch.cat((emb1, emb2, emb3,emb4), dim=1)
        cx1 = self.attn_norm1(emb1)
        cx2 = self.attn_norm2(emb2)
        cx3 = self.attn_norm3(emb3)
        cx4 = self.attn_norm4(emb4)
        emb_all = self.attn_norm(emb_all)
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = org1 + cx1
        cx2 = org2 + cx2
        cx3 = org3 + cx3
        cx4 = org4 + cx4

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1)
        x2 = self.ffn_norm2(cx2)
        x3 = self.ffn_norm3(cx3)
        x4 = self.ffn_norm4(cx4)
        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)
        x3 = self.ffn3(x3)
        x4 = self.ffn4(x4)
        x1 = x1 + org1
        x2 = x2 + org2
        x3 = x3 + org3
        x4 = x4 + org4

        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1)
        emb2 = self.encoder_norm2(emb2)
        emb3 = self.encoder_norm3(emb3)
        emb4 = self.encoder_norm4(emb4)
        return emb1, emb2, emb3, emb4, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[8, 16, 32, 64], patchSize=[16, 8, 4, 2]):
        super().__init__()

        self.embeddings_1 = Channel_Embeddings(config, patchSize[0], img_size=img_size, in_channels=channel_num[0]) # 16, 256, 8
        self.embeddings_2 = Channel_Embeddings(config, patchSize[1], img_size=img_size // 2, in_channels=channel_num[1]) # 8, 128, 16
        self.embeddings_3 = Channel_Embeddings(config, patchSize[2], img_size=img_size // 4, in_channels=channel_num[2]) # 4, 64, 32
        self.embeddings_4 = Channel_Embeddings(config, patchSize[3], img_size=img_size // 8, in_channels=channel_num[3]) # 2, 32, 64

        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(patchSize[0], patchSize[0])) # 8 16
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(patchSize[1], patchSize[1])) # 16 8
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(patchSize[2], patchSize[2])) # 32 4
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(patchSize[3], patchSize[3])) # 64 2

    def forward(self, en1, en2, en3, en4):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1, emb2, emb3, emb4)  # (B, n_patch, hidden)

        x1 = self.reconstruct_1(encoded1)
        x2 = self.reconstruct_2(encoded2)
        x3 = self.reconstruct_3(encoded3)
        x4 = self.reconstruct_4(encoded4)

        x1 = x1 + en1
        x2 = x2 + en2
        x3 = x3 + en3
        x4 = x4 + en4

        return x1, x2, x3, x4, attn_weights

# up of SCTransNet starts here
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)



def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels, F_x=in_channels)
        self.nConvs = _make_nConv(in_channels * 2, out_channels, nb_Conv, activation) # because of concat

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


def _make_layer(block, input_channels, output_channels, num_blocks=1):
    layers = []
    layers.append(block(input_channels, output_channels))
    for i in range(num_blocks - 1):
        layers.append(block(output_channels, output_channels))
    return nn.Sequential(*layers)


class DSTransNet_Student(nn.Module):
    def __init__(self, config, input_channels=1, n_classes=2, img_size=256, vis=False, mode='train', deepsuper=True):
        
        super().__init__()
        self.config = config
        self.input_channels = input_channels # 1
        self.n_classes = n_classes # 2
        self.img_size = img_size # 256
        self.vis = vis # False
        self.mode = mode # train
        self.deepsuper = deepsuper # True
        self.base_channels = config.base_channels  # 8

        self.block = Res_block
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.input_head = _make_layer(self.block, self.input_channels, self.base_channels * 1) # 1 -> 8 [1 8 256 256]
        self.down_encoder1 = _make_layer(self.block, self.base_channels * 1, self.base_channels * 2, 1)  # 8 -> 16 [1 16 128 128 ]
        self.down_encoder2 = _make_layer(self.block, self.base_channels * 2, self.base_channels * 4, 1)  # 16 -> 32 [1 32 64 64 ]
        self.down_encoder3 = _make_layer(self.block, self.base_channels * 4, self.base_channels * 8, 1)  # 32 -> 64 [1 64 32 32]
        self.down_encoder4 = _make_layer(self.block, self.base_channels * 8, self.base_channels * 8, 1)  # 64 -> 64 [1 64 16 16]


        # Skip Connection
        self.mtc = ChannelTransformer(self.config, self.vis, self.img_size, channel_num=[self.base_channels * 1, self.base_channels * 2, self.base_channels * 4, self.base_channels * 8], patchSize=config.patch_sizes)
        

        # Decoder
        self.up_decoder4 = UpBlock_attention(self.base_channels * 8,  self.base_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(self.base_channels * 4,  self.base_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(self.base_channels * 2,  self.base_channels * 1, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(self.base_channels * 1,  self.base_channels * 1, nb_Conv=2)
        self.output_head = nn.Conv2d(self.base_channels, self.n_classes, kernel_size=(1, 1), stride=(1, 1))

        # Deep Supervision
        if self.deepsuper:
            self.gt_conv4 = nn.Sequential(nn.Conv2d(self.base_channels * 8, self.n_classes, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(self.base_channels * 4, self.n_classes, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(self.base_channels * 2, self.n_classes, 1))
            self.gt_conv1 = nn.Sequential(nn.Conv2d(self.base_channels * 1, self.n_classes, 1))
            self.gt_conv0 = nn.Sequential(nn.Conv2d(5 * self.n_classes, self.n_classes, 1))


        # Adapter Layer
        self.student_base_channels = self.base_channels
        self.teacher_base_channels = 32

        self.x1_conv = nn.Conv2d(in_channels = self.student_base_channels * 1, out_channels = self.teacher_base_channels * 1, kernel_size=1)
        self.x2_conv = nn.Conv2d(in_channels = self.student_base_channels * 2, out_channels = self.teacher_base_channels * 2, kernel_size=1)
        self.x3_conv = nn.Conv2d(in_channels = self.student_base_channels * 4, out_channels = self.teacher_base_channels * 4, kernel_size=1)
        self.x4_conv = nn.Conv2d(in_channels = self.student_base_channels * 8, out_channels = self.teacher_base_channels * 8, kernel_size=1)

        self.skip_x1_conv = nn.Conv2d(in_channels = self.student_base_channels * 1, out_channels = self.teacher_base_channels * 1, kernel_size=1)
        self.skip_x2_conv = nn.Conv2d(in_channels = self.student_base_channels * 2, out_channels = self.teacher_base_channels * 2, kernel_size=1)
        self.skip_x3_conv = nn.Conv2d(in_channels = self.student_base_channels * 4, out_channels = self.teacher_base_channels * 4, kernel_size=1)
        self.skip_x4_conv = nn.Conv2d(in_channels = self.student_base_channels * 8, out_channels = self.teacher_base_channels * 8, kernel_size=1)

        self.d1_conv = nn.Conv2d(in_channels = self.student_base_channels * 1, out_channels = self.teacher_base_channels * 1, kernel_size=1)
        self.d2_conv = nn.Conv2d(in_channels = self.student_base_channels * 1, out_channels = self.teacher_base_channels * 1, kernel_size=1)
        self.d3_conv = nn.Conv2d(in_channels = self.student_base_channels * 2, out_channels = self.teacher_base_channels * 2, kernel_size=1)
        self.d4_conv = nn.Conv2d(in_channels = self.student_base_channels * 4, out_channels = self.teacher_base_channels * 4, kernel_size=1)

    def forward(self, x):

        # Encoder
        x1 = self.input_head(x)  # x1: 1 8 256 256
        x2 = self.down_encoder1(self.pool(x1))  # x2: 1 16 128 128
        x3 = self.down_encoder2(self.pool(x2))  # x3: 1 32 64 64
        x4 = self.down_encoder3(self.pool(x3))  # x4: 1 64 32 32
        d5 = self.down_encoder4(self.pool(x4))  # d5: 1 64 16 16


        # DAT Module
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4


        # DRA & Decoder
        d4 = self.up_decoder4(d5, x4) # [1, 64, 16, 16] + [1, 64, 32, 32] -> [1, 32, 32, 32] 
        d3 = self.up_decoder3(d4, x3) # [1, 32, 32, 32] + [1, 32, 64, 64] -> [1, 16, 64, 64]
        d2 = self.up_decoder2(d3, x2) # [1, 16, 64, 64] + [1, 16, 128, 128] -> [1, 8, 128, 128]
        d1 = self.up_decoder1(d2, x1) # [1, 8, 128, 128] + [1, 8, 256, 256] -> [1, 8, 256, 256]
        out = self.output_head(d1)  # [1, 8, 256, 256] -> [1, 1, 256, 256]

        # deep supervision
        if self.deepsuper:
            gt_4 = self.gt_conv4(d5)
            gt_3 = self.gt_conv3(d4)
            gt_2 = self.gt_conv2(d3)
            gt_1 = self.gt_conv1(d2)

            gt_4 = F.interpolate(gt_4, scale_factor=16, mode='nearest')
            gt_3 = F.interpolate(gt_3, scale_factor=8, mode='nearest')
            gt_2 = F.interpolate(gt_2, scale_factor=4, mode='nearest')
            gt_1 = F.interpolate(gt_1, scale_factor=2, mode='nearest')

            gt_0 = self.gt_conv0(torch.cat((gt_4, gt_3, gt_2, gt_1, out), 1))

            if self.mode == 'train':
                return [d5, d4, d3, d2, d1], (gt_4, gt_3, gt_2, gt_1, out, gt_0)
            else:
                return out
        else:
            return out


class DSTransNet_Student_v34(nn.Module):
    def __init__(self, config, input_channels=1, n_classes=2, img_size=256, vis=False, mode='train', deepsuper=True):
        
        super().__init__()
        self.config = config
        self.input_channels = input_channels # 1
        self.n_classes = n_classes # 2
        self.img_size = img_size # 256
        self.vis = vis # False
        self.mode = mode # train
        self.deepsuper = deepsuper # True
        self.base_channels = config.base_channels  # 8

        self.block = Res_block
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.input_head = _make_layer(self.block, self.input_channels, self.base_channels * 1) # 1 -> 8 [1 8 256 256]
        self.down_encoder1 = _make_layer(self.block, self.base_channels * 1, self.base_channels * 2, 1)  # 8 -> 16 [1 16 128 128 ]
        self.down_encoder2 = _make_layer(self.block, self.base_channels * 2, self.base_channels * 4, 1)  # 16 -> 32 [1 32 64 64 ]
        self.down_encoder3 = _make_layer(self.block, self.base_channels * 4, self.base_channels * 8, 1)  # 32 -> 64 [1 64 32 32]
        self.down_encoder4 = _make_layer(self.block, self.base_channels * 8, self.base_channels * 8, 1)  # 64 -> 64 [1 64 16 16]


        # Skip Connection
        self.mtc = ChannelTransformer(self.config, self.vis, self.img_size, channel_num=[self.base_channels * 1, self.base_channels * 2, self.base_channels * 4, self.base_channels * 8], patchSize=config.patch_sizes)
        

        # Decoder
        self.up_decoder4 = UpBlock_attention(self.base_channels * 8,  self.base_channels * 4, nb_Conv=2)
        self.up_decoder3 = UpBlock_attention(self.base_channels * 4,  self.base_channels * 2, nb_Conv=2)
        self.up_decoder2 = UpBlock_attention(self.base_channels * 2,  self.base_channels * 1, nb_Conv=2)
        self.up_decoder1 = UpBlock_attention(self.base_channels * 1,  self.base_channels * 1, nb_Conv=2)
        self.output_head = nn.Conv2d(self.base_channels, self.n_classes, kernel_size=(1, 1), stride=(1, 1))

        # Deep Supervision
        if self.deepsuper:
            self.gt_conv4 = nn.Sequential(nn.Conv2d(self.base_channels * 8, self.n_classes, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(self.base_channels * 4, self.n_classes, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(self.base_channels * 2, self.n_classes, 1))
            self.gt_conv1 = nn.Sequential(nn.Conv2d(self.base_channels * 1, self.n_classes, 1))
            self.gt_conv0 = nn.Sequential(nn.Conv2d(5 * self.n_classes, self.n_classes, 1))


        # Adapter Layer
        self.student_base_channels = self.base_channels
        self.teacher_base_channels = 32

        self.x1_conv = nn.Conv2d(in_channels = self.student_base_channels * 1, out_channels = self.teacher_base_channels * 1, kernel_size=1)
        self.x2_conv = nn.Conv2d(in_channels = self.student_base_channels * 2, out_channels = self.teacher_base_channels * 2, kernel_size=1)
        self.x3_conv = nn.Conv2d(in_channels = self.student_base_channels * 4, out_channels = self.teacher_base_channels * 4, kernel_size=1)
        self.x4_conv = nn.Conv2d(in_channels = self.student_base_channels * 8, out_channels = self.teacher_base_channels * 8, kernel_size=1)

        self.skip_x1_conv = nn.Conv2d(in_channels = self.student_base_channels * 1, out_channels = self.teacher_base_channels * 1, kernel_size=1)
        self.skip_x2_conv = nn.Conv2d(in_channels = self.student_base_channels * 2, out_channels = self.teacher_base_channels * 2, kernel_size=1)
        self.skip_x3_conv = nn.Conv2d(in_channels = self.student_base_channels * 4, out_channels = self.teacher_base_channels * 4, kernel_size=1)
        self.skip_x4_conv = nn.Conv2d(in_channels = self.student_base_channels * 8, out_channels = self.teacher_base_channels * 8, kernel_size=1)

        self.d1_conv = nn.Conv2d(in_channels = self.student_base_channels * 1, out_channels = self.teacher_base_channels * 1, kernel_size=1)
        self.d2_conv = nn.Conv2d(in_channels = self.student_base_channels * 1, out_channels = self.teacher_base_channels * 1, kernel_size=1)
        self.d3_conv = nn.Conv2d(in_channels = self.student_base_channels * 2, out_channels = self.teacher_base_channels * 2, kernel_size=1)
        self.d4_conv = nn.Conv2d(in_channels = self.student_base_channels * 4, out_channels = self.teacher_base_channels * 4, kernel_size=1)

    def forward(self, x):

        # Encoder
        x1 = self.input_head(x)  # x1: 1 8 256 256
        x2 = self.down_encoder1(self.pool(x1))  # x2: 1 16 128 128
        x3 = self.down_encoder2(self.pool(x2))  # x3: 1 32 64 64
        x4 = self.down_encoder3(self.pool(x3))  # x4: 1 64 32 32
        d5 = self.down_encoder4(self.pool(x4))  # d5: 1 64 16 16


        # DAT Module
        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        x1, x2, x3, x4, att_weights = self.mtc(x1, x2, x3, x4)
        x1 = x1 + f1
        x2 = x2 + f2
        x3 = x3 + f3
        x4 = x4 + f4


        # DRA & Decoder
        d4 = self.up_decoder4(d5, x4) # [1, 64, 16, 16] + [1, 64, 32, 32] -> [1, 32, 32, 32] 
        d3 = self.up_decoder3(d4, x3) # [1, 32, 32, 32] + [1, 32, 64, 64] -> [1, 16, 64, 64]
        d2 = self.up_decoder2(d3, x2) # [1, 16, 64, 64] + [1, 16, 128, 128] -> [1, 8, 128, 128]
        d1 = self.up_decoder1(d2, x1) # [1, 8, 128, 128] + [1, 8, 256, 256] -> [1, 8, 256, 256]
        out = self.output_head(d1)  # [1, 8, 256, 256] -> [1, 1, 256, 256]

        # deep supervision
        if self.deepsuper:
            gt_4 = self.gt_conv4(d5)
            gt_3 = self.gt_conv3(d4)
            gt_2 = self.gt_conv2(d3)
            gt_1 = self.gt_conv1(d2)

            gt_4 = F.interpolate(gt_4, scale_factor=16, mode='nearest')
            gt_3 = F.interpolate(gt_3, scale_factor=8, mode='nearest')
            gt_2 = F.interpolate(gt_2, scale_factor=4, mode='nearest')
            gt_1 = F.interpolate(gt_1, scale_factor=2, mode='nearest')

            gt_0 = self.gt_conv0(torch.cat((gt_4, gt_3, gt_2, gt_1, out), 1))

            if self.mode == 'train':
                return [d5, d4, d3, d2, d1], (gt_4, gt_3, gt_2, gt_1, out, gt_0)
            else:
                return out
        else:
            return out


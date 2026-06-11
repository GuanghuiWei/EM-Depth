# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DepthDecoder_My2S(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, youtube_pretrain=False, is_test=False):
        super(DepthDecoder_My2S, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = num_ch_enc        #features in encoder, [64, 18, 36, 72, 144]    [64, 32, 64, 128, 256]
        self.use_bilinear = not youtube_pretrain and is_test

        # decoder
        self.convs = OrderedDict()

        self.convs[("parallel_conv"), 0, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 0, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 0, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        self.convs[("parallel_conv"), 0, 4] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
        self.convs[("conv1x1", 0, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.convs[("conv1x1", 0, 3_2)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.convs[("conv1x1", 0, 3_1)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[1])
        self.convs[("conv1x1", 0, 4_3)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[3])
        self.convs[("conv1x1", 0, 4_2)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[2])
        self.convs[("conv1x1", 0, 4_1)] = ConvBlock1x1(self.num_ch_enc[4], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 1, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 1, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("parallel_conv"), 1, 3] = ConvBlock(self.num_ch_enc[3], self.num_ch_enc[3])
        self.convs[("conv1x1", 1, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.convs[("conv1x1", 1, 3_2)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.convs[("conv1x1", 1, 3_1)] = ConvBlock1x1(self.num_ch_enc[3], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 2, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("parallel_conv"), 2, 2] = ConvBlock(self.num_ch_enc[2], self.num_ch_enc[2])
        self.convs[("conv1x1", 2, 2_1)] = ConvBlock1x1(self.num_ch_enc[2], self.num_ch_enc[1])

        self.convs[("parallel_conv"), 3, 0] = ConvBlock(self.num_ch_enc[0], self.num_ch_enc[0])
        self.convs[("parallel_conv"), 3, 1] = ConvBlock(self.num_ch_enc[1], self.num_ch_enc[1])
        self.convs[("conv1x1", 3, 1_0)] = ConvBlock1x1(self.num_ch_enc[1], self.num_ch_enc[0])

        self.convs[("parallel_conv"), 4, 0] = ConvBlock(self.num_ch_enc[0], 32)
        self.convs[("parallel_conv"), 5, 0] = ConvBlock(32, 16)
        # self.convs[("parallel_conv"), 5, 1] = Down_Conv3x3(16, 16)
        # self.convs[("parallel_conv"), 5, 2] = Down_Conv3x3(16, 16)
        # self.convs[("parallel_conv"), 5, 3] = Down_Conv3x3(16, 16)

        self.convs[("dispconv", 0)] = Conv3x3(16, self.num_output_channels)
        # self.convs[("dispconv", 1)] = Conv3x3(16, self.num_output_channels)
        # self.convs[("dispconv", 2)] = Conv3x3(16, self.num_output_channels)
        # self.convs[("dispconv", 3)] = Conv3x3(16, self.num_output_channels)
        self.convs[("AdaptiveWeighting_4", 0, 1)] = AdaptiveWeighting_4(self.num_ch_enc[1], ratio=2) #features in encoder, [64, 18, 36, 72, 144] [64, 32, 64, 128, 256]

        self.convs[("AdaptiveWeighting_3", 0, 2)] = AdaptiveWeighting_3(self.num_ch_enc[2], ratio=3) # [ 64  64 128 256 512]
        self.convs[("AdaptiveWeighting_2", 0, 3)] = AdaptiveWeighting_2(self.num_ch_enc[3], ratio=4)

        self.convs[("AdaptiveWeighting_3", 1, 1)] = AdaptiveWeighting_3(self.num_ch_enc[1], ratio=2)
        self.convs[("AdaptiveWeighting_2", 1, 2)] = AdaptiveWeighting_2(self.num_ch_enc[2], ratio=3)

        self.convs[("AdaptiveWeighting_2", 2, 1)] = AdaptiveWeighting_2(self.num_ch_enc[1], ratio=2)

        self.convs[("AdaptiveWeighting_2", 3, 0)] = AdaptiveWeighting_2(self.num_ch_enc[0], ratio=4)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, input_features):
        self.outputs = {}

        # features in encoder
        e4 = input_features[4]
        e3 = input_features[3]
        e2 = input_features[2]
        e1 = input_features[1]
        e0 = input_features[0]

        d0_1 = self.convs[("parallel_conv"), 0, 1](e1)
        d0_2 = self.convs[("parallel_conv"), 0, 2](e2)
        d0_3 = self.convs[("parallel_conv"), 0, 3](e3)
        d0_4 = self.convs[("parallel_conv"), 0, 4](e4)

        d0_2_1 = updown_sample(d0_2, 2)
        d0_3_2 = updown_sample(d0_3, 2)
        d0_3_1 = updown_sample(d0_3, 4)
        d0_4_3 = updown_sample(d0_4, 2)
        d0_4_2 = updown_sample(d0_4, 4)
        d0_4_1 = updown_sample(d0_4, 8)

        d0_2_1 = self.convs[("conv1x1", 0, 2_1)](d0_2_1)
        d0_3_2 = self.convs[("conv1x1", 0, 3_2)](d0_3_2)
        d0_3_1 = self.convs[("conv1x1", 0, 3_1)](d0_3_1)
        d0_4_3 = self.convs[("conv1x1", 0, 4_3)](d0_4_3)
        d0_4_2 = self.convs[("conv1x1", 0, 4_2)](d0_4_2)
        d0_4_1 = self.convs[("conv1x1", 0, 4_1)](d0_4_1)

        # d0_1_msf = d0_1 + d0_2_1 + d0_3_1 + d0_4_1
        # d0_2_msf = d0_2 + d0_3_2 + d0_4_2
        # d0_3_msf = d0_3 + d0_4_3

        d0_1_msf = self.convs[("AdaptiveWeighting_4", 0, 1)](d0_1, d0_2_1, d0_3_1, d0_4_1)
        d0_2_msf = self.convs[("AdaptiveWeighting_3", 0, 2)](d0_2, d0_3_2, d0_4_2)
        d0_3_msf = self.convs[("AdaptiveWeighting_2", 0, 3)](d0_3, d0_4_3)

        d1_1 = self.convs[("parallel_conv"), 1, 1](d0_1_msf)
        d1_2 = self.convs[("parallel_conv"), 1, 2](d0_2_msf)
        d1_3 = self.convs[("parallel_conv"), 1, 3](d0_3_msf)

        d1_2_1 = updown_sample(d1_2, 2)
        d1_3_2 = updown_sample(d1_3, 2)
        d1_3_1 = updown_sample(d1_3, 4)

        d1_2_1 = self.convs[("conv1x1", 1, 2_1)](d1_2_1)
        d1_3_2 = self.convs[("conv1x1", 1, 3_2)](d1_3_2)
        d1_3_1 = self.convs[("conv1x1", 1, 3_1)](d1_3_1)
        #
        # d1_1_msf = d1_1 + d1_2_1 + d1_3_1
        # d1_2_msf = d1_2 + d1_3_2
        #
        d1_1_msf = self.convs[("AdaptiveWeighting_3", 1, 1)](d1_1, d1_2_1, d1_3_1)
        d1_2_msf = self.convs[("AdaptiveWeighting_2", 1, 2)](d1_2, d1_3_2)

        d2_1 = self.convs[("parallel_conv"), 2, 1](d1_1_msf)
        d2_2 = self.convs[("parallel_conv"), 2, 2](d1_2_msf)

        d2_2_1 = updown_sample(d2_2, 2)

        d2_2_1 = self.convs[("conv1x1", 2, 2_1)](d2_2_1)

        # d2_1_msf = d2_1 + d2_2_1
        d2_1_msf = self.convs[("AdaptiveWeighting_2", 2, 1)](d2_1, d2_2_1)

        d3_0 = self.convs[("parallel_conv"), 3, 0](e0)
        d3_1 = self.convs[("parallel_conv"), 3, 1](d2_1_msf)

        if self.use_bilinear:
            d3_1_0 = up_sample(d3_1, 2)
        else:
            d3_1_0 = updown_sample(d3_1, 2)

        d3_1_0 = self.convs[("conv1x1", 3, 1_0)](d3_1_0)
        #
        # d3_0_msf = d3_0 + d3_1_0
        d3_0_msf = self.convs[("AdaptiveWeighting_2", 3, 0)](d3_0, d3_1_0)

        d4_0 = self.convs[("parallel_conv"), 4, 0](d3_0_msf)

        if self.use_bilinear:
            d4_0 = up_sample(d4_0, 2)
        else:
            d4_0 = updown_sample(d4_0, 2)

        d5 = self.convs[("parallel_conv"), 5, 0](d4_0)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](d5))
        #self.outputs[("disp", 0)] = self.relu(self.convs[("dispconv", 0)](d5))

        return self.outputs     #single-scale depth

class AdaptiveWeighting_4(nn.Module):
    def __init__(self, channels, ratio):
        super(AdaptiveWeighting_4, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, x3, x4):
        batch, channels, _, _ = x1.size()

        pooled = self.global_pool(x1 + x2 + x3 + x4).view(batch, channels)
        weights = self.fc(pooled)
        weights = weights * 4
        w1, w2, w3, w4 = weights[:, 0].view(batch, 1, 1, 1), weights[:, 1].view(batch, 1, 1, 1), \
                        weights[:, 2].view(batch, 1, 1, 1), weights[:, 3].view(batch, 1, 1, 1)

        return w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4

class AdaptiveWeighting_3(nn.Module):
    def __init__(self, channels, ratio):
        super(AdaptiveWeighting_3, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, 3),
            nn.Softmax(dim=1)
        )
    def forward(self, x1, x2, x3):
        batch, channels, _, _ = x1.size()

        pooled = self.global_pool(x1 + x2 + x3).view(batch, channels)
        weights = self.fc(pooled)
        weights = weights * 3
        w1, w2, w3 = weights[:, 0].view(batch, 1, 1, 1), weights[:, 1].view(batch, 1, 1, 1), weights[:, 2].view(batch, 1, 1, 1)

        return w1 * x1 + w2 * x2 + w3 * x3


class AdaptiveWeighting_2(nn.Module):
    def __init__(self, channels, ratio):
        super(AdaptiveWeighting_2, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(),
            nn.Linear(channels // ratio, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, high_res, low_res):
        batch, channels, _, _ = high_res.size()

        pooled = self.global_pool(high_res + low_res).view(batch, channels)
        weights = self.fc(pooled)
        weights = weights * 2
        high_weight, low_weight = weights[:, 0].view(batch, 1, 1, 1), weights[:, 1].view(batch, 1, 1, 1)

        return high_res * high_weight + low_res * low_weight
        

class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out
        
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=3 // 2, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
        
          
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def updown_sample(x, scale_fac):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode="nearest")

def up_sample(x, scale_fac):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode="bilinear")
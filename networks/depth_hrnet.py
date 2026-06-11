from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict
#from layers import *
from timm.models.layers import trunc_normal_
from .hrnet import hrnet18
from .hrnet import hrnet48


class DepthEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_layer=None):#, is_feature=False
        super(DepthEncoder, self).__init__()
        assert num_layers == 18 or 48

        # self.is_feature = is_feature
        if num_layers == 18:
            self.encoder = hrnet18(pretrained)
        if num_layers == 48:
            self.encoder = hrnet48(pretrained)

        self.num_ch_enc = self.encoder.num_ch_enc

    def forward(self, x):
        x = (x - 0.45) / 0.225
        self.features = self.encoder(x)

        return self.features



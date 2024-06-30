# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn

from mmseg.ops import resize
from ..builder import NECKS


@NECKS.register_module()
class SegFormerNeck(nn.Module):

    def __init__(self, 
                 in_channels=[512],
                 out_channel=1024):
        super(SegFormerNeck, self).__init__()
        self.out_channel=out_channel
        self.in_channels=in_channels
        self.dim_cat=lambda x,shape:torch.cat(
            [
                resize(
                    input=each_x,
                    size=shape,
                    mode='bilinear',
                    align_corners=False,
                )
                for each_x in x
            ],
            dim=1
        )
        self.dim_fuse=nn.Conv2d(
            in_channels=sum(self.in_channels),
            out_channels=self.out_channel,
            kernel_size=1,
        )

        #self.test_init()

    def test_init(self):
        for name,param in self.named_parameters():
            param.requires_grad=False

    def forward(self, x):
        size= x[-2].shape[-2:]  # 因为倒数第二个的scale_factor是1
        x = self.dim_cat(x,size)
        x = self.dim_fuse(x)
        return x

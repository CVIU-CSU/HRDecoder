import torch
import torch.nn as nn

from mmseg.ops import resize
from ..builder import NECKS

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

@NECKS.register_module()
class SAMNeck(nn.Module):
    def __init__(self,
                 dim=256,
                 out_channels=[64,128,320,512],
                 use_conv=True,
                 scale_factors=[4,2,1,0.5],
                 norm_layer=LayerNorm):
        super(SAMNeck,self).__init__()
        self.dim=dim
        self.out_channels=out_channels
        self.use_conv=use_conv
        self.scale_factors=scale_factors
        self.norm_layer = norm_layer

        self.stages = nn.ModuleList()
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            out_channel = self.out_channels[idx]
            if scale == 8.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    self.norm_layer(dim//2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    self.norm_layer(dim//4),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
                ]
                out_dim = dim // 8
            elif scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    self.norm_layer(dim//2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    nn.Conv2d(
                        out_dim,
                        out_channel,
                        kernel_size=1,
                        bias=False,
                    ),
                    self.norm_layer(out_channel),
                    nn.Conv2d(
                        out_channel,
                        out_channel,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    ),
                    self.norm_layer(out_channel),
                ]
            )
            layers = nn.Sequential(*layers)
            self.stages.append(layers)

    def forward(self,x):
        feature=[]
        for stage in self.stages:
            feature.append(stage(x))
        return feature

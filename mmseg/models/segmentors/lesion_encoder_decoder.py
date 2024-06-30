import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize

from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class LesionEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 use_sigmoid=False,
                 my_metrics=False,
                 crop_info_path=None,
                 pretrained=None,
                 **kwargs):
        super(EncoderDecoder, self).__init__()
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        
        if backbone.type == 'ConvNeXt':
            from mmpretrain.models import ConvNeXt
            backbone.pop('type')
            self.backbone = ConvNeXt(**backbone)
        else:
            self.backbone = builder.build_backbone(backbone)
        
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # NEW: use sigmoid activation
        self.use_sigmoid = use_sigmoid
        self.my_metrics = my_metrics

        assert self.with_decode_head

    def forward_train(self, 
                      img, 
                      img_metas, 
                      gt_semantic_seg, 
                      seg_weight=None, 
                      return_feat=False,
                      **kwargs,):
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x

        logit = self.decode_head(x)
        seg_loss = self.decode_head.losses(logit,gt_semantic_seg,seg_weight)
        losses.update(add_prefix(seg_loss,'decode'))
        
        if kwargs.pop('return_logit',False):
            losses['logit']=logit

        return losses

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)

        # NEW:
        if img_meta[0]['pad_shape'] != img_meta[0]['img_shape']:
            img_shape = img_meta[0]['img_shape']
            seg_logit = seg_logit[:, :, :img_shape[0], :img_shape[1]]

        if rescale:
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole', 'whole_lip']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        elif self.test_cfg.mode == 'whole_lip':
            seg_logit = self.whole_inference_lip(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        # output = F.softmax(seg_logit, dim=1)
        output = logit_activation(seg_logit, self.use_sigmoid)  # use sigmoid or softmax
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)

        if self.use_sigmoid or self.my_metrics:
            try:
                compute_aupr = self.test_cfg.compute_aupr
            except AttributeError:
                compute_aupr = False

            if compute_aupr:
                seg_logit = seg_logit.squeeze(0).cpu().numpy()
                seg_logit = [(seg_logit, self.use_sigmoid, compute_aupr)]
            else:
                seg_logit = (seg_logit > 0.5).int()
                seg_logit = seg_logit.squeeze(0).cpu().numpy()
            return seg_logit

        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


def logit_activation(seg_logit, use_sigmoid=False):
    """
    :param seg_logit: feature map without activation function
    :param use_sigmoid: whether to use sigmoid
    :return: activation feature map
    """

    if not use_sigmoid:
        output = F.softmax(seg_logit, dim=1)
    else:
        output = torch.sigmoid(seg_logit)

    return output
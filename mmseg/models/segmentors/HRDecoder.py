import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import SEGMENTORS,build_head
from ..decode_heads.segformer_head import MLP
from copy import deepcopy
from .lesion_encoder_decoder import LesionEncoderDecoder


def get_crop_bbox(img_size,hr_scale,divisible=1):
    assert hr_scale[0] > 0 and hr_scale[1] > 0
    if img_size==hr_scale:
        return (0, img_size[0], 0, img_size[1])
    margin_h = max(img_size[0] - hr_scale[0], 0)
    margin_w = max(img_size[1] - hr_scale[1], 0)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + hr_scale[0]
    crop_x1, crop_x2 = offset_w, offset_w + hr_scale[1]

    return crop_y1, crop_y2, crop_x1, crop_x2

def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img

@SEGMENTORS.register_module()
class HRDecoder(LesionEncoderDecoder):
    def __init__(self,hr_settings,**kwargs):
        super(HRDecoder,self).__init__(**kwargs)
        self._init_hr_settings(hr_settings,kwargs)

    def _init_hr_settings(self, hr_settings, kwargs):
        self.hr_scale=hr_settings.hr_scale
        self.divisible=hr_settings.get('divisible',8)
        self.hr_loss_weight=hr_settings.get('hr_loss_weight',0.1)
        self.lr_loss_weight=hr_settings.get('lr_loss_weight',0)
        self.crop_num = hr_settings.pop('crop_num', 4)
        self.scale_ratio = hr_settings.pop('scale_ratio', 1)

    def _transform_inputs(self, inputs):
        upsampled_inputs = [
            resize(x,inputs[0].shape[2:],mode='bilinear') 
            for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        return inputs
    
    def get_random_hr_scale(self):
        if isinstance(self.scale_ratio, (tuple, list)):
            min_ratio, max_ratio = self.scale_ratio
            assert min_ratio <= max_ratio
            ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
            hr_scale = (int(self.hr_scale[0]*ratio//self.divisible*self.divisible),
                        int(self.hr_scale[1]*ratio//self.divisible*self.divisible))
            return hr_scale
        else:
            return self.hr_scale

    def _resizecrop_feats(self, ori_size, feat, bbox):
        h_grids, w_grids = ori_size[0]//feat.shape[-2], ori_size[1]//feat.shape[-1] # h,w
        lr_bbox = [bbox[0]//h_grids, bbox[1]//h_grids, bbox[2]//w_grids, bbox[3]//w_grids]
        lr_crop_feat = crop(feat, lr_bbox)
        hr_feat = resize(lr_crop_feat, feat.shape[2:], mode='bilinear')
        return hr_feat
    
    def hr_forward_train(self, lr_feat, gt_semantic_seg):
        img_size = gt_semantic_seg.shape[2:]
        hr_scale = self.get_random_hr_scale()
        hr_bboxes = [get_crop_bbox(img_size,hr_scale,self.divisible) for i in range(self.crop_num)]
        hr_labels = torch.cat([crop(gt_semantic_seg, hr_bbox) for hr_bbox in hr_bboxes])
        hr_feats = torch.cat([self._resizecrop_feats(img_size, lr_feat, hr_bbox) for hr_bbox in hr_bboxes])
        
        hr_logits = self.decode_head([hr_feats])
        hr_losses = self.decode_head.losses(hr_logits, hr_labels)
        hr_losses['loss_seg'] = hr_losses['loss_seg'] * self.hr_loss_weight
        return hr_losses

    def hr_slide_inference(self,ori_size, lr_feats, img_metas):
        h_stride, w_stride = h_crop, w_crop = self.hr_scale
        h_img, w_img = ori_size
        bs = lr_feats.shape[0]
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_boxes = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1, x1 = h_idx * h_stride, w_idx * w_stride
                y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                crop_boxes.append([y1, y2, x1, x2]) 

        crop_feats = torch.cat([self._resizecrop_feats(ori_size,lr_feats,bbox) for bbox in crop_boxes])
        crop_logits = self.decode_head([crop_feats])
        crop_logits = resize(crop_logits,self.hr_scale,mode='bilinear')\
            .reshape(-1,bs,self.num_classes,*self.hr_scale)
        
        return {'logits': crop_logits, 'feats':crop_feats, 'boxes': crop_boxes}
    
    def encode_decode(self, img, img_metas):
        img_size=img.shape[2:]
        lr_img = resize(img, self.hr_scale,mode='bilinear')
        lr_feat = self._transform_inputs(self.extract_feat(lr_img))
        lr_logit = self._decode_head_forward_test([lr_feat],img_metas)
        up_lr_logit = resize(lr_logit, img_size, mode='bilinear')
        
        fuse_logit = up_lr_logit.clone()
        for logit,_,bbox in zip(*self.hr_slide_inference(img_size, lr_feat, img_metas).values()):
                slc = slice(bbox[0],bbox[1]),slice(bbox[2],bbox[3])
                fuse_logit[:,:,slc[0],slc[1]] = (logit + up_lr_logit[:,:,slc[0],slc[1]])/2
        
        fuse_logit = resize(fuse_logit, img_size, mode='bilinear')
        return fuse_logit

    def forward_train(self, 
                      img, 
                      img_metas, 
                      gt_semantic_seg, 
                      seg_weight=None, 
                      return_feat=False):
        img_size=img.shape[2:]
        losses=dict()

        lr_img = resize(img,self.hr_scale,mode='bilinear')
        lr_feat = self._transform_inputs(self.extract_feat(lr_img))
        lr_logit = self.decode_head([lr_feat])
        if self.lr_loss_weight>0:
            lr_loss = self.decode_head.losses(lr_logit, gt_semantic_seg)
            lr_loss['loss_seg'] = lr_loss['loss_seg'] * self.lr_loss_weight
            losses.update(add_prefix(lr_loss, 'lr'))
            
        hr_losses = self.hr_forward_train(lr_feat,gt_semantic_seg)
        losses.update(add_prefix(hr_losses,'hr'))

        up_lr_logit = resize(lr_logit, img_size, mode='bilinear')
        fuse_logit = up_lr_logit.clone()
        for logit,_,bbox in zip(*self.hr_slide_inference(img_size, lr_feat, img_metas).values()):
            slc = slice(bbox[0],bbox[1]),slice(bbox[2],bbox[3])
            fuse_logit[:,:,slc[0],slc[1]] = (logit + up_lr_logit[:,:,slc[0],slc[1]])/2
        
        fuse_loss = self.decode_head.losses(fuse_logit,gt_semantic_seg)
        losses.update(add_prefix(fuse_loss,'fuse'))

        return losses

        

@SEGMENTORS.register_module()
class EfficientHRDecoder(HRDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _init_hr_settings(self, hr_settings, kwargs):
        in_channels = hr_settings.pop('in_channels', 720)  # sum([48,96,192,384])
        visual_dim = hr_settings.pop('visual_dim',256)
        from mmcv.cnn import ConvModule
        self.multiscale_compressor = ConvModule(
            in_channels = in_channels,
            out_channels= visual_dim,
            kernel_size=1,
            conv_cfg=self.decode_head.conv_cfg,
            norm_cfg=self.decode_head.norm_cfg,
            act_cfg=self.decode_head.act_cfg,
        )
        super()._init_hr_settings(hr_settings, kwargs)

    def _transform_inputs(self, inputs):
        return self.multiscale_compressor(super(EfficientHRDecoder,self)._transform_inputs(inputs))

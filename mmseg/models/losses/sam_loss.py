import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

def cross_entropy(input,
                  target,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        input,
        target,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def softCrossEntropy(input, target, reduction='mean'):
    log_logit = -F.log_softmax(input, dim=1)
    batch = input.shape[0]
    if reduction == 'batchmean':
        loss = torch.mean(torch.mul(log_logit, target)) / batch
    elif reduction=='mean':
        loss = torch.mean(torch.mul(log_logit, target))
    elif reduction=='batchsum':
        loss = torch.sum(torch.mul(log_logit, target)) / batch
    elif reduction=='sum':
        loss = torch.sum(torch.mul(log_logit, target))
    else:
        loss = torch.mul(log_logit, target)
    return loss


@LOSSES.register_module()
class SAMLoss(nn.Module):
    def __init__(self,
                 size_average=None,
                 reduce=None,
                 use_kl=True,
                 one_hot=False,
                 reduction='batchmean',
                 log_target=False,
                 loss_weight=1,
                 ):
        super(SAMLoss,self).__init__()
        self.size_average=size_average
        self.reduce=reduce
        self.use_kl=use_kl
        self.one_hot=one_hot
        self.reduction=reduction
        self.log_target=log_target
        self.loss_weight=loss_weight
        if self.use_kl:
            self.loss_func = F.kl_div 
        elif self.one_hot:
            self.loss_func = cross_entropy
        else:
            self.loss_func = softCrossEntropy
        

    def forward(self,
                seg_logit,
                gt_logit,
                seg_weight=None,
                ):
        seg_logit=torch.log_softmax(seg_logit,dim=1)#.cpu()
        #gt_logit=gt_logit.cpu()
        if self.use_kl:
            loss = self.loss_weight* self.loss_func(
                input=seg_logit,
                target=gt_logit,
                size_average=self.size_average,
                reduce=self.reduce,
                reduction=self.reduction,
                log_target=self.log_target,
            )
        elif self.one_hot:
            gt_logit=gt_logit.squeeze(1)
            loss = self.loss_weight*self.loss_func(
                input=seg_logit,
                target=gt_logit,
                weight=seg_weight,
            )
        else:
            loss = self.loss_weight* self.loss_func(
                input=seg_logit,
                target=gt_logit,
                reduction=self.reduction,
            )
        #loss = loss.to(seg_logit.device)
        return loss

# REF: https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def _make_one_hot(gt, num_classes, ignore=(0, 255)):
    """
    :param label: [N, *], values in [0,num_classes)
    :param ignore: ignore value of background, here is (0, 255)
    :return: [N, C, *]
    """
    # label = gt.clone()
    label = gt
    label = label.unsqueeze(1)
    shape = list(label.shape)
    shape[1] = num_classes + 1

    if ignore is not None:
        if 0 in ignore:
            for index in ignore:
                label[label == index] = num_classes + 1
            label = label - 1
        else:
            for index in ignore:
                label[label == index] = num_classes

    result = torch.zeros(shape, device=label.device)

    # print(set(label.detach().cpu().numpy().flatten().tolist())) # for DEBUG
    result.scatter_(1, label, 1)

    return result[:, :-1, ]


def binary_ce_loss(pred, label, **kwargs):
    loss = F.binary_cross_entropy(pred, label, reduction='none')
    loss = torch.mean(loss, dim=(1, 2))
    return loss


def binary_dice_loss(pred, label, smooth=1.0, exponent=2):
    """
    :param pred: [N, *]: here should be scores in [0,1]
    :param label: [N, *]: values in [0,1]
    :param smooth: smooth
    :return: [N]
    """

    pred = pred.contiguous().view(pred.shape[0], -1).float()
    label = label.contiguous().view(label.shape[0], -1).float()

    num = 2 * torch.sum(torch.mul(pred, label), dim=1) + smooth
    den = torch.sum(pred.pow(exponent) + label.pow(exponent), dim=1) + smooth

    loss = 1. - num / den

    return loss


def binary_ce_dice_loss(pred, label, smooth=1.0, **kwargs):
    loss1 = binary_ce_loss(pred, label, **kwargs)
    loss2 = binary_dice_loss(pred, label, smooth=smooth)

    return loss1 + loss2


def binary_loss(pred_raw,
                label_raw,
                loss_func,
                weight=None,
                class_weight=None,
                class_weight_norm=False,
                reduction='mean',
                avg_factor=None,
                smooth=1.0,
                ODOC=False,
                **kwargs):
    """
    :param pred:  [N, C, *] scores without softmax
    :param label: [N, *] in [0, C], 0 stands for background, 1~C stands for pred in 0~C-1
    :return: reduction([N])
    """
    pred = pred_raw.clone()
    label = label_raw.clone()
    num_classes = pred.shape[1]
    if class_weight is not None:
        class_weight = class_weight.float()

    if pred.shape != label.shape:
        label = _make_one_hot(label, num_classes)

    pred = torch.sigmoid(pred)

    loss = 0.
    for i in range(num_classes):
        if isinstance(loss_func, tuple):
            loss_function = loss_func[i]
        else:
            loss_function = loss_func
        if i == 0 and ODOC:
            label[:, i] = (label != 0).any(axis=1)
        class_loss = loss_function(pred[:, i], label[:, i], smooth=smooth)
        if class_weight is not None:
            class_loss *= class_weight[i]
        loss += class_loss

    if class_weight is not None and class_weight_norm:
        loss = loss / torch.sum(class_weight)
    else:
        loss = loss / num_classes

    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class BinaryLoss(nn.Module):
    def __init__(self,
                 loss_type='ce',
                 reduction='mean',
                 class_weight=None,
                 class_weight_norm=False,
                 loss_weight=1.0,
                 smooth=1.0,
                 ODOC=False,
                 loss_name='loss_bde',
                 **kwargs):
        super(BinaryLoss, self).__init__()
        assert loss_type in ['ce', 'dice', 'ce_dice','tversky','focaltversky','dice_focaltversky']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.class_weight_norm = class_weight_norm
        self.loss_type = loss_type
        self.smooth = smooth
        self.ODOC = ODOC
        self._loss_name = loss_name

        self.loss_hyper = kwargs.get('loss_hyper',None)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
            assert class_weight.shape[0] == cls_score.shape[1], \
                'Expect weight shape [{}], get[{}]'.format(cls_score.shape[1], class_weight.shape[0])
        else:
            class_weight = None

        loss_func = None
        if self.loss_type == 'ce':
            loss_func = binary_ce_loss
        elif self.loss_type == 'dice':
            loss_func = binary_dice_loss
        elif self.loss_type == 'ce_dice':
            loss_func = binary_ce_dice_loss
        elif self.loss_type == 'tversky':
            loss_func = TverskyLoss(**self.loss_hyper)
        elif self.loss_type == 'focaltversky':
            loss_func = FocalTverskyLoss(**self.loss_hyper)
        elif self.loss_type == 'dice_focaltversky':
            loss_func = DiceFocalTverskyLoss(**self.loss_hyper)

        loss_cls = self.loss_weight * binary_loss(
            cls_score,
            label,
            loss_func,
            weight,
            class_weight=class_weight,
            class_weight_norm=self.class_weight_norm,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            ODOC=self.ODOC
        )
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    


def tversky_loss(pred, label, alpha=0.5, beta=0.5, smooth=1.0, exponent=2.0):
    pred = pred.contiguous().view(pred.shape[0], -1).float()
    label = label.contiguous().view(label.shape[0], -1).float()

    num = torch.sum(torch.mul(pred, label), dim=1)
    fn = torch.sum(torch.mul(1-pred,label), dim=1)
    fp = torch.sum(torch.mul(pred,1-label), dim=1)

    loss = 1. - (num + smooth) / (num + alpha*fn + beta*fp + smooth)
    return loss


def focal_tversky_loss(pred, label, alpha=0.5, beta=0.5, gamma=0.75, smooth=1.0, exponent=2.0):
    return tversky_loss(pred,label,alpha,beta,smooth,exponent).pow(gamma)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = 1-alpha

    def forward(self,pred,label,smooth=1.0,exponent=2):
        return tversky_loss(
            pred=pred,
            label=label,
            alpha=self.alpha,
            beta=self.beta,
            smooth=smooth,
            exponent=exponent,
        )

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.75) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = 1-alpha
        self.gamma = gamma

    def forward(self,pred,label,smooth=1.0,exponent=2):
        return binary_dice_loss(pred=pred,label=label,smooth=smooth) + \
            focal_tversky_loss(
                pred=pred,
                label=label,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                smooth=smooth,
                exponent=exponent,
            )
    
class DiceFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.75) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = 1-alpha
        self.gamma = gamma

    def forward(self,pred,label,smooth=1.0,exponent=2):
        return (
            binary_dice_loss(
                pred=pred,
                label=label,
                smooth=smooth) 
            + focal_tversky_loss(
                pred=pred,
                label=label,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                smooth=smooth,
                exponent=exponent,
            )
        )/2
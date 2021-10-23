import os
import numpy as np
import torch
import torch.nn as nn

from utils_junjie.iou_mask import iou_rle, bboxes_iou
import models.losses
import torch.nn.functional as F


class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr


def ConvBnLeaky(in_, out_, k, s):
    '''
    in_: input channel, e.g. 32
    out_: output channel, e.g. 64
    k: kernel size, e.g. 3 or (3,3)
    s: stride, e.g. 1 or (1,1)
    '''
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_, out_, k, s, padding=pad, bias=False),
        nn.BatchNorm2d(out_, eps=1e-5, momentum=0.1),
        nn.LeakyReLU(0.1)
        # nn.ReLU()
    )


class EmptyLayer(nn.Module):  # 只是为了占位，以便处理route层和shortcut层
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode

    def forward(self, x):
        # x = F.interpolate(x, size=[self.scale[0], self.scale[1]], mode='nearest')
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode)
        return x


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, alpha=1.5, gamma=0.25, reduction="mean"):
        super(FocalLoss, self).__init__()
        # apply focal loss to each element
        loss_fcn.reduction = "none"
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        # print("Use focal loss, gamma:", gamma)

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1 + 1e-6 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == "none":
            return loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


class YOLOBranch(nn.Module):
    '''
    Calculate the output boxes and losses.
    '''

    def __init__(self, all_anchors, anchor_indices, num_classes,  **kwargs):
        super().__init__()
        # self.anchors_all = all_anchors
        self.anchor_indices = anchor_indices
        self.anchors = all_anchors[anchor_indices]
        # anchors: tensor, e.g. shape(2,3), [[116,90],[156,198]]
        self.num_anchors = len(anchor_indices)
        self.num_classes = num_classes
        # all anchors, (0, 0, w, h), used for calculating IoU
        self.anch_00wha_all = torch.zeros(len(all_anchors), 4)
        self.anch_00wha_all[:, 2:4] = all_anchors  # absolute, degree

        self.ignore_thre = 0.6
        self.obj_scale = 1
        self.noobj_scale = 5
        # self.loss4obj = FocalLoss(nn.BCELoss(), reduction='sum')
        self.loss4obj = nn.BCELoss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        loss_angle = kwargs.get('loss_angle', 'period_L1')
        if loss_angle in {'L1', 'LL1'}:
            self.loss4angle = nn.L1Loss(reduction='sum')
        elif loss_angle in {'L2', 'LL2'}:
            self.loss4angle = nn.MSELoss(reduction='sum')
        elif loss_angle == 'BCE':
            self.loss4angle = nn.BCELoss(reduction='sum')
        elif loss_angle == 'period_L1':
            self.loss4angle = models.losses.period_L1(reduction='sum')
        elif loss_angle == 'period_L2':
            self.loss4angle = models.losses.period_L2(reduction='sum')
        elif loss_angle == 'none':
            # inference
            pass
        else:
            raise Exception('unknown loss for angle')
        self.laname = loss_angle
        self.angle_range = kwargs.get('angran', 360)
        assert self.angle_range in {180, 360}

    def forward(self, raw, img_size, labels=None):
        """
        Args:
            raw: tensor with shape [batchsize, anchor_num*6, size, size]
            img_size: int, image resolution
            labels: ground truth annotations
        """
        assert raw.dim() == 4
        # TODO: for now we require the input to be a square though it's not necessary.

        # raw
        device = raw.device
        nB = raw.shape[0]  # batch size
        nA = self.num_anchors  # number of anchors
        nCH = self.num_classes  # number of channels, 6=(x,y,w,h,angle,conf)

        raw = raw.view(nB, nA, nCH + 4, raw.shape[2], raw.shape[3])
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()

        # sigmoid activation for xy, angle, obj_conf
        xy_offset = torch.sigmoid(raw[..., 0:2])  # x,y
        # linear activation for w, h
        wh_scale = raw[..., 2:4]
        conf = torch.sigmoid(raw[..., 4:])
        # cls = torch.sigmoid(raw[..., 5:])  # Cls pred.

        # calculate pred - xywh obj cls
        x_shift = torch.arange(raw.shape[3], dtype=torch.float, device=device
                               ).repeat(raw.shape[2], 1).view(1, 1, raw.shape[2], raw.shape[3])
        y_shift = torch.arange(raw.shape[2], dtype=torch.float, device=device
                               ).repeat(raw.shape[3], 1).t().view(1, 1, raw.shape[2], raw.shape[3])

        # NOTE: anchors are not normalized
        anchors = self.anchors.clone().to(device=device)
        anch_w = anchors[:, 0].view(1, nA, 1, 1)  # absolute
        anch_h = anchors[:, 1].view(1, nA, 1, 1)  # absolute

        pred_final = torch.ones(nB, nA, raw.shape[2], raw.shape[3], nCH + 4, device=device)
        pred_final[..., 0] = (xy_offset[..., 0] + x_shift) / raw.shape[3]  # normalized 0-1
        pred_final[..., 1] = (xy_offset[..., 1] + y_shift) / raw.shape[2]  # normalized 0-1
        pred_final[..., 2] = torch.exp(wh_scale[..., 0]) * anch_w  # absolute
        pred_final[..., 3] = torch.exp(wh_scale[..., 1]) * anch_h  # absolute
        pred_final[..., 4:] = conf
        # pred_final[..., 5:] = cls

        if labels is None:
            # inference, convert final predictions to absolute
            pred_final[..., 0] *= img_size[3]
            pred_final[..., 1] *= img_size[2]
            return pred_final.view(nB, -1, nCH + 4).detach(), None
        else:
            # training, convert final predictions to be normalized
            pred_final[..., 2] /= img_size[3]
            pred_final[..., 3] /= img_size[2]
            # force the normalized w and h to be <= 1
            pred_final[..., 0:4].clamp_(min=0, max=1)

        pred_boxes = pred_final[..., :4].detach()  # xywh normalized, a degree
        pred_confs = pred_final[..., 4:].detach()
        # pred_cls = pred_final[..., 5:].detach()

        # target assignment
        obj_mask = torch.zeros(nB, nA, raw.shape[2], raw.shape[3], dtype=torch.bool, device=device)
        noobj_mask = torch.ones(nB, nA, raw.shape[2], raw.shape[3], dtype=torch.bool, device=device)
        penalty_mask = torch.ones(nB, nA, raw.shape[2], raw.shape[3], dtype=torch.bool, device=device)
        target = torch.zeros(nB, nA, raw.shape[2], raw.shape[3], nCH+4, dtype=torch.float, device=device)

        labels = labels.detach()
        nlabel = (labels[:, :, 1:5].sum(dim=2) > 0).sum(dim=1)  # number of objects
        labels = labels.to(device=device)

        tx_all, ty_all = labels[:, :, 1] * raw.shape[3], labels[:, :, 2] * raw.shape[2]  # 0-nG
        tw_all, th_all = labels[:, :, 3], labels[:, :, 4]  # normalized 0-1

        ti_all = tx_all.long()
        tj_all = ty_all.long()

        norm_anch_wh = anchors[:, 0:2]
        norm_anch_wh[:, 0] /= img_size[3]
        norm_anch_wh[:, 1] /= img_size[2]
        norm_anch_00wha = self.anch_00wha_all.clone().to(device=device)
        norm_anch_00wha[:, 2] /= img_size[3]  # normalized
        norm_anch_00wha[:, 3] /= img_size[2]

        # traverse all images in a batch
        valid_gt_num = 0
        for b in range(nB):
            label_mask = torch.zeros(labels.shape[1], dtype=torch.bool, device=device)
            n = int(nlabel[b])  # number of ground truths in b'th image
            if n == 0:
                # no ground truth
                continue
            gt_boxes = torch.zeros(n, 4, device=device)
            gt_boxes[:, 2] = tw_all[b, :n]  # normalized 0-1
            gt_boxes[:, 3] = th_all[b, :n]  # normalized 0-1

            anchor_ious = bboxes_iou(gt_boxes, norm_anch_00wha, xyxy=False)
            best_n_all = torch.argmax(anchor_ious, dim=1)
            best_n = best_n_all % self.num_anchors

            valid_mask = torch.zeros(n, dtype=torch.bool, device=device)
            for ind in self.anchor_indices.to(device=device):
                valid_mask = (valid_mask | (best_n_all == ind))
            if sum(valid_mask) == 0:
                # no anchor is responsible for any ground truth
                continue
            else:
                valid_gt_num += sum(valid_mask)
            label_mask[:n] = valid_mask
            best_n = best_n[valid_mask]
            try:
                truth_i = ti_all[b, :n][valid_mask]
                truth_j = tj_all[b, :n][valid_mask]
            except Exception as e:
                print("INFO:.....")
            gt_boxes[:, 0] = tx_all[b, :n] / raw.shape[3]  # normalized 0-1
            gt_boxes[:, 1] = ty_all[b, :n] / raw.shape[2]  # normalized 0-1

            # print(torch.cuda.memory_allocated()/1024/1024/1024, 'GB')
            # gt_boxes e.g. shape(11,4)
            selected_idx = pred_confs[b].max(-1)[0] > 0.5
            selected = pred_boxes[b][selected_idx]
            if len(selected) < 2000 and len(selected) > 0:
                # ignore some predicted boxes who have high overlap with any groundtruth
                pred_ious = iou_rle(selected.view(-1, 4), gt_boxes, xywha=True,
                                    is_degree=True, img_size=img_size, normalized=True)
                pred_best_iou, _ = pred_ious.max(dim=1)
                to_be_ignored = (pred_best_iou > self.ignore_thre)
                # set mask to zero (ignore) if the pred BB has a large IoU with any gt BB
                # penalty_mask[b, selected_idx] = ~to_be_ignored
                noobj_mask[b, selected_idx] = ~to_be_ignored

            # penalty_mask[b, best_n, truth_j, truth_i] = 1
            try:
                obj_mask[b, best_n, truth_j, truth_i] = 1
                noobj_mask[b, best_n, truth_j, truth_i] = 0
                target[b, best_n, truth_j, truth_i, 0] = tx_all[b, :n][valid_mask] - tx_all[b, :n][valid_mask].floor()
            except Exception as e:
                print("INFO:.....")
            target[b, best_n, truth_j, truth_i, 1] = ty_all[b, :n][valid_mask] - ty_all[b, :n][valid_mask].floor()
            target[b, best_n, truth_j, truth_i, 2] = torch.log(
                tw_all[b, :n][valid_mask] / norm_anch_wh[best_n, 0] + 1e-16)
            target[b, best_n, truth_j, truth_i, 3] = torch.log(
                th_all[b, :n][valid_mask] / norm_anch_wh[best_n, 1] + 1e-16)

            target[b, best_n, truth_j, truth_i, labels[b][label_mask][:, 0].type(torch.int64) + 4] = 1  # objectness confidence
            # target[b, best_n, truth_j, truth_i, labels[b][label_mask][:, 0].type(torch.int64) + 5] = 1

        self.loss_xy = self.bce_loss(xy_offset[obj_mask], target[..., 0:2][obj_mask])
        wh_pred = wh_scale[obj_mask]
        wh_target = target[..., 2:4][obj_mask]
        self.loss_wh = self.l2_loss(wh_pred, wh_target)
        self.loss_obj = self.loss4obj(conf[obj_mask], target[..., 4:][obj_mask])
        self.loss_noobj = self.loss4obj(conf[noobj_mask], target[..., 4:][noobj_mask])

        loss = self.loss_xy + 0.5 * self.loss_wh + self.loss_obj + self.loss_noobj
        ngt = valid_gt_num + 1e-16
        self.gt_num = valid_gt_num
        self.loss_str = 'total {} objects: '.format(int(ngt)) +\
                        'xy/gt {}, wh/gt {}'.format('%.3f'%(self.loss_xy/ngt), '%.3f'%(self.loss_wh/ngt)) +\
                        'conf {}'.format('%.3f'%self.loss_obj)

        return None, loss
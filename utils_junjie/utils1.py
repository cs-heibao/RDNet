from PIL import Image
import random
import torch
from torchvision import transforms

from utils_junjie.iou_mask import iou_rle, bbox_iou
import copy
import tqdm
import cv2
import numpy as np


def normalize_bbox(xywha, w, h, max_angle=1):
    '''
    Normalize bounding boxes to 0~1 range

    Args:
        xywha: torch.tensor, bounding boxes, shape(...,5)
        w: image width
        h: image height
        max_angle: the angle will be divided by max_angle
    '''
    assert torch.is_tensor(xywha)

    if xywha.dim() == 1:
        assert xywha.shape[0] == 5
        xywha[0] /= w
        xywha[1] /= h
        xywha[2] /= w
        xywha[3] /= h
        xywha[4] /= max_angle
    elif xywha.dim() == 2:
        # assert xywha.shape[1] == 5
        xywha[:,1] /= w
        xywha[:,2] /= h
        xywha[:,3] /= w
        xywha[:,4] /= h
        # xywha[:,4] /= max_angle
    else:
        raise Exception('unkown bbox format')
    
    return xywha

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def rect_to_square(img, labels, target_size, pro_type, pad_value=0, aug=False):
    '''
    Pre-processing during training and testing

    1. Resize img such that the longer side of the image = target_size;
    2. Pad the img it to square

    Arguments:
        img: PIL image
        labels: torch.tensor, shape(N,5), [cx, cy, w, h, angle], not normalized
        target_size: int, e.g. 608
        pad_value: int
        aug: bool
    '''
    # assert isinstance(img, Image.Image)
    if pro_type == 'torch':
        ori_h, ori_w = img.height, img.width
    else:
        ori_h, ori_w = img.shape[0], img.shape[1]
    # resize to target input size (usually smaller)
    resize_scale = max(target_size) / max(ori_w, ori_h)
    # unpad_w, unpad_h = target_size * w / max(w,h), target_size * h / max(w,h)
    unpad_w, unpad_h = int(ori_w * resize_scale), int(ori_h * resize_scale)
    # pad to square
    if aug:
        # random placing
        left = random.randint(0, target_size[1] - unpad_w)
        if (target_size[0] - unpad_h)<0:
            print(target_size, unpad_w, unpad_h)
        top = random.randint(0, target_size[0] - unpad_h)
    else:
        left = (target_size[1] - unpad_w) // 2
        top = (target_size[0] - unpad_h) // 2
    right = target_size[1] - unpad_w - left
    bottom = target_size[0] - unpad_h - top
    # record the padding info
    img_tl = (left, top)  # start of the true image
    # img_tl = (0, 0)  # start of the true image
    img_wh = (unpad_w, unpad_h)
    if pro_type == 'torch':
        img = transforms.functional.resize(img, (unpad_h, unpad_w))
        img = transforms.functional.pad(img, padding=(left, top, right, bottom), fill=0)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (unpad_w, unpad_h))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # modify labels
    if labels is not None:
        # labels[:, 0:4] *= resize_scale
        labels[:, 1:5:2] *= float(unpad_w/ori_w)
        labels[:, 2:5:2] *= float(unpad_h/ori_h)
        labels[:, 1] += left
        labels[:, 2] += top

    pad_info = torch.Tensor((ori_w, ori_h) + img_tl + img_wh)
    return img, labels, pad_info


def detection2original(boxes, pad_info):
    '''
    Recover the bbox from the resized and padded image to the original image.

    Args:
        boxes: tensor, rows of [cx, cy, w, h, angle]
        pad_info: (ori w, ori h, tl x, tl y, imw, imh)
    '''
    assert boxes.dim() == 2
    ori_w, ori_h, tl_x, tl_y, imw, imh = pad_info
    boxes[:,0] = (boxes[:,0] - tl_x) / imw * ori_w
    boxes[:,1] = (boxes[:,1] - tl_y) / imh * ori_h
    boxes[:,2] = (boxes[:,2] - tl_x) / imw * ori_w
    boxes[:,3] = (boxes[:,3] - tl_y) / imh * ori_h

    return boxes
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    # for image_i, image_pred in enumerate(prediction):
    # Filter out confidence scores below threshold
    # image_pred = prediction[prediction[:, 4] >= conf_thres]
    # If none are remaining => process next image
    # if not image_pred.size[0]:
    #     continue
    # Object confidence times class confidence
    score = prediction[:, 4:].max(1)[0]
    # Sort by it
    image_pred = prediction[(-score).argsort()]
    class_confs, class_preds = image_pred[:, 4:].max(1, keepdim=True)
    detections = torch.cat((image_pred[:, :4], class_confs.float(), class_preds.float()), 1)
    # Perform non-maximum suppression
    keep_boxes = []
    while detections.size(0):
        large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
        label_match = detections[0, -1] == detections[:, -1]
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        weights = detections[invalid, 4:5]
        # Merge overlapping bboxes by order of confidence
        detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
        keep_boxes += [detections[0]]
        detections = detections[~invalid]
    if keep_boxes:
        output = torch.stack(keep_boxes)
    # y = output.new(output.shape)
    # y[..., 0] = (output[..., 0] + output[..., 2]) / 2
    # y[..., 1] = (output[..., 1] + output[..., 3]) / 2
    # y[..., 2] = output[..., 2] - output[..., 0]
    # y[..., 3] = output[..., 3] - output[..., 1]
    # y[..., 4:] = output[..., 4:]
    return output

def nms(detections, is_degree=True, nms_thres=0.45, img_size=2048):
    '''
    Single-class non-maximum suppression for bounding boxes with angle.
    
    Args:
        detections: rows of (x,y,w,h,angle,conf,...)
        is_degree: True -> input angle is degree, False -> radian
        nms_thres: suppresion IoU threshold
        img_size: int, preferably the image size
    '''
    assert (detections.dim() == 2) and (detections.shape[1] >= 5)
    device = detections.device
    if detections.shape[0] == 0:
        return detections
    # sort by confidence
    idx = torch.argsort(detections[:,4], descending=True)
    detections = detections[idx,:]

    boxes = detections[:,0:4] # only [x,y,w,h,a]
    boxes_angle = boxes.clone()
    boxes_angle[:,-1] = 0
    valid = torch.zeros(boxes.shape[0], dtype=torch.bool, device=device)
    # the first one is always valid
    valid[0] = True
    # only one candidate at the beginning. Its votes number is 1 (it self)
    votes = [1]
    for i in range(1, boxes.shape[0]):
        # compute IoU with valid boxes
        ious = iou_rle(boxes[i], boxes[valid,:], xywha=True, is_degree=is_degree,
                      img_size=img_size)

        # the i'th BB is invalid if it is similar to any valid BB
        if (ious >= nms_thres).any():
            continue
        # else, this box is valid
        valid[i] = True
        # the votes number of the new candidate BB is 1 (it self)
        votes.append(1)

    selected = detections[valid,:]
    return selected

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    # for sample_i in range(len(outputs)):
    if outputs.shape[0] ==0:
        return batch_metrics
    output = outputs
    pred_boxes = output[:, :4]
    pred_scores = output[:, 4]
    pred_labels = output[:, -1]

    true_positives = np.zeros(pred_boxes.shape[0])

    annotations = targets
    target_labels = annotations[:, 0] if len(annotations) else []
    if len(annotations):
        detected_boxes = []
        target_boxes = annotations[:, 1:]

        for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

            # If targets are found break
            if len(detected_boxes) == len(annotations):
                break

            # Ignore if label is not one of the target labels
            if pred_label not in target_labels:
                continue

            iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
            if iou >= iou_threshold and box_index not in detected_boxes:
                true_positives[pred_i] = 1
                detected_boxes += [box_index]
    batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_per_class(tp, conf, pred_cls, target_cls, rec_thres):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            # the original false/true positive calculation exits problem
            # fpc = (1 - tp[i]).cumsum()
            # tpc = (tp[i]).cumsum()
            fpc = (1 - tp).cumsum()
            tpc = (tp).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            ap.append(compute_ap(recall_curve, precision_curve))

    # return p, r, REcurve, PRcurve, unique_classes.astype("int32")
    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

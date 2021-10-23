import os
import json
import random
from PIL import Image
import numpy as np
from collections import defaultdict

import torch
import torchvision.transforms.functional as tvf

from utils_junjie.utils import *
import utils_junjie.augmentation as augUtils
import cv2
from math import pi
import copy
# import imgaug.augmenters as iaa
# from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import math
import numpy as np

# class DefaultAug():
#     def __init__(self, ):
#         self.augmentations = iaa.Sequential([
#             iaa.Dropout([0.0, 0.01]),
#             iaa.Sharpen((0.0, 0.2)),
#             iaa.Affine(rotate=(-20, 20), translate_percent=(-0.2,0.2)),  # rotate by -45 to 45 degrees (affects segmaps)
#             iaa.AddToBrightness((-30, 30)),
#             iaa.AddToHue((-20, 20)),
#             iaa.Fliplr(0.5),
#         ], random_order=True)
#
#     def aug(self, img, boxes):
#         # Unpack data
#         # img, boxes = data
#
#         # Convert xywh to xyxy
#         boxes = np.array(boxes)
#         boxes[:, 1:] = self.xywh2xyxy_np(boxes[:, 1:])
#
#         # Convert bounding boxes to imgaug
#         bounding_boxes = BoundingBoxesOnImage(
#             [BoundingBox(*box[1:], label=box[0]) for box in boxes],
#             shape=img.shape)
#
#         # Apply augmentations
#         img, bounding_boxes = self.augmentations(
#             image=img,
#             bounding_boxes=bounding_boxes)
#
#         # Clip out of image boxes
#         bounding_boxes = bounding_boxes.clip_out_of_image()
#
#         # Convert bounding boxes back to numpy
#         boxes = np.zeros((len(bounding_boxes), 5))
#         for box_idx, box in enumerate(bounding_boxes):
#             # Extract coordinates for unpadded + unscaled image
#             x1 = box.x1
#             y1 = box.y1
#             x2 = box.x2
#             y2 = box.y2
#
#             # Returns (x, y, w, h)
#             boxes[box_idx, 0] = box.label
#             boxes[box_idx, 1] = ((x1 + x2) / 2)
#             boxes[box_idx, 2] = ((y1 + y2) / 2)
#             boxes[box_idx, 3] = (x2 - x1)
#             boxes[box_idx, 4] = (y2 - y1)
#
#         return img, boxes
#
#     def xywh2xyxy_np(self, x):
#         y = np.zeros_like(x)
#         y[..., 0] = x[..., 0] - x[..., 2] / 2
#         y[..., 1] = x[..., 1] - x[..., 3] / 2
#         y[..., 2] = x[..., 0] + x[..., 2] / 2
#         y[..., 3] = x[..., 1] + x[..., 3] / 2
#         return y

class Dataset4YoloAngle(torch.utils.data.Dataset):
    """
    dataset class.
    """
    def __init__(self, img_dir, json_path, labels, img_size=(512, 512), augmentation=False, mosaic=False, pro_type='torch', hpy=None):
        """
        dataset initialization. Annotation data are read into memory by API.

        Args:
            img_dir: str or list, imgs folder, e.g. 'someDir/COCO/train2017/'
            json_path: str or list, e.g. 'someDir/COCO/instances_train2017.json'
            img_size: int, target image size input to the YOLO, default: 608
            augmentation: bool, default: True
            only_person: bool, if true, non-person BBs are discarded. default: True
            debug: bool, if True, only one data id is selected from the dataset
        """
        self.labels = labels
        self.pro_type = pro_type
        self.max_labels = 50
        self.img_size = img_size
        self.enable_aug = augmentation
        self.hpy = hpy
        self.mosaic = mosaic
        self.mosaic_border = [-img_size[0] // 2, -img_size[1] // 2]
        self.img_ids = []
        self.imgid2info = dict()
        self.imgid2path = dict()
        self.imgid2anns = defaultdict(list)
        self.catids = []
        if isinstance(img_dir, str):
            assert isinstance(json_path, str)
            img_dir, json_path = [img_dir], [json_path]
        assert len(img_dir) == len(json_path)
        for imdir,jspath in zip(img_dir, json_path):
            self.load_anns(imdir, jspath)
        print('\nTraining Datasets are: {}'.format(len(self.img_ids)))
        random.shuffle(self.img_ids)

    def load_anns(self, img_dir, json_path):
        '''
        laod json file to self.img_ids, self.imgid2anns
        '''
        self.coco = False
        print('Loading annotations {} into memory...'.format(json_path))
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for ann in json_data['annotations']:
            if ann['category_id'] + 1 in self.labels.keys():
                img_id = ann['image_id']
                if len(ann['bbox'])==0:
                    self.imgid2anns[img_id] = []
                else:
                    ann['bbox'] = torch.Tensor(ann['bbox'])

                # ann['bbox'] = torch.Tensor(ann['bbox'])
                # rad = ann['bbox'][-1] * pi / 180
                # width = ann['bbox'][2] * torch.cos(rad) + ann['bbox'][3] * torch.abs(torch.sin(rad))
                # height = ann['bbox'][2] * torch.abs(torch.sin(rad)) + ann['bbox'][3] * torch.cos(rad)
                # ann['bbox'][2] = width
                # ann['bbox'][3] = height
                # ann['bbox'] = ann['bbox'][:-1]

                self.imgid2anns[img_id].append(ann)
        for img in json_data['images']:
            img_id = img['id']
            if img_id in self.imgid2anns.keys():
                try:
                    assert img_id not in self.imgid2path
                except Exception as e:
                    print(img_id)
                anns = self.imgid2anns[img_id]
                # if there is crowd gt, skip this image
                if self.coco and any(ann['iscrowd'] for ann in anns):
                    continue
                # # if only for person detection
                # if self.only_person:
                #     # select the images which contain at least one person
                #     if not any(ann['category_id']==1 for ann in anns):
                #         continue
                #     # and ignore all other categories
                #     self.imgid2anns[img_id] = [a for a in anns if a['category_id']==1]
                self.img_ids.append(img_id)
                self.imgid2path[img_id] = os.path.join(img_dir, img['file_name'])
                self.imgid2info[img['id']] = img
        # shuffle datasets
        random.shuffle(self.img_ids)
        self.catids = [cat['id'] for cat in json_data['categories']]
        # self.catids = [1, 2]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
        index (int): data index
        """
        # here add more augmentations
        hpy = self.hpy
        if self.mosaic and random.random()<hpy['mosaic']:
            input_img, label = self.load_mosaic(index)
            label = torch.Tensor(label)
            nL = len(label)  # number of labels
            labels = torch.zeros(self.max_labels, 5)
            labels[:label.shape[0], ...] = label
        else:
            input_img, labels = self.load_image(index)
            nL = len(labels.sum(axis=1) != 0)
            # convert xywh2xyxy
            labels[:, 1:5] = xywh2xyxy(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] *= input_img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] *= input_img.shape[1]
        if self.enable_aug:
            if not self.mosaic and random.random()<hpy['mosaic']:
                input_img, labels = random_perspective(input_img, labels,
                                                 degrees=hpy['degrees'],
                                                 translate=hpy['translate'],
                                                 scale=hpy['scale'],
                                                 shear=hpy['shear'],
                                                 perspective=hpy['perspective'])
            # Augment colorspace
            augment_hsv(input_img, hgain=hpy['hsv_h'], sgain=hpy['hsv_s'], vgain=hpy['hsv_v'])
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= input_img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= input_img.shape[1]  # normalized width 0-1

        # make sure after augmentation, the box's width or height is not zero
        labels_new = torch.zeros(self.max_labels, 5)
        non_zero = (labels[:, -1] * labels[:, -2]!=0).sum()
        labels_new[:non_zero, ...] = labels[labels[:, -1] * labels[:, -2]!=0]

        # input_img = np.ascontiguousarray(input_img)

        if self.pro_type=='torch':
            input_ori = tvf.to_tensor(input_img)
            img = input_ori
            # img = input_ori.unsqueeze(0)
        else:
            input_ori = torch.from_numpy(input_img).permute((2, 0, 1)).float() / 255.
            img = input_ori
            # img = input_ori.unsqueeze(0)

        # target = labels
        # np_img = (img.data.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        # np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        # boxes = target.numpy()
        # for box in boxes:
        #     if sum(box) == 0:
        #         continue
        #     x = box[1] * np_img.shape[1]
        #     y = box[2] * np_img.shape[0]
        #     w = box[3] * np_img.shape[1]
        #     h = box[4] * np_img.shape[0]
        #     x1 = x - w / 2
        #     y1 = y - h / 2
        #     x2 = x + w / 2
        #     y2 = y + h / 2
        #     cv2.rectangle(np_img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 255), thickness=1,
        #                   lineType=cv2.LINE_4)
        # cv2.namedWindow('1', cv2.WINDOW_NORMAL)
        # cv2.imshow('1', np_img)
        # cv2.waitKey()

        return img, labels_new


    def augment_PIL(self, img, labels):
        if np.random.rand() > 0.4:
            img = tvf.adjust_brightness(img, uniform(0.3,1.5))
        if np.random.rand() > 0.7:
            factor = 2 ** uniform(-1, 1)
            img = tvf.adjust_contrast(img, factor) # 0.5 ~ 2
        # if np.random.rand() > 0.7:
        #     img = tvf.adjust_hue(img, uniform(-0.1,0.1))
        # if np.random.rand() > 0.6:
        #     factor = uniform(0,2)
        #     if factor > 1:
        #         factor = 1 + uniform(0, 2)
        #     img = tvf.adjust_saturation(img, factor) # 0 ~ 3
        if np.random.rand() > 0.5:
            img = tvf.adjust_gamma(img, uniform(0.5, 3))
        # # horizontal flip
        # if np.random.rand() > 0.5:
        #     img, labels = augUtils.hflip(img, labels)
        # vertical flip
        #if np.random.rand() > 0.5:
        #    img, labels = augUtils.vflip(img, labels)
        # # random rotation
        #rand_degree = np.random.rand() * 360
        #if self.coco:
        #    img, labels = augUtils.rotate(img, rand_degree, labels, expand=True)
        #else:
        #    img, labels = augUtils.rotate(img, rand_degree, labels, expand=False)
        return img, labels


    def load_mosaic(self, index):
        # loads images in a mosaic

        labels4 = []
        sh, sw = self.img_size[0], self.img_size[1]
        # yc, xc = [int(random.uniform(-x, 2 * sh + x)) for x in self.mosaic_border]  # mosaic center x, y
        yc = int(random.uniform(-self.mosaic_border[0], 2 * sh + self.mosaic_border[0]))  # mosaic center x, y
        xc = int(random.uniform(-self.mosaic_border[1], 2 * sw + self.mosaic_border[1]))  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, x = self.load_image(index)

            # sh, sw = img.shape[0], img.shape[1]
            # self.mosaic_border = [-img.shape[0] // 2, -img.shape[1] // 2]
            #
            # # yc, xc = [int(random.uniform(-x, 2 * sh + x)) for x in self.mosaic_border]  # mosaic center x, y
            # yc = int(random.uniform(-self.mosaic_border[0], 2 * sh + self.mosaic_border[0]))  # mosaic center x, y
            # xc = int(random.uniform(-self.mosaic_border[1], 2 * sw + self.mosaic_border[1]))  # mosaic center x, y

            x = x[x.sum(axis=1)!=0]
            h = img.shape[0]
            w = img.shape[1]
            # img, _, (h, w) = self.load_image(self, index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((sh * 2, sw * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, sw * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(sh * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, sw * 2), min(sh * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            # x = self.labels[index]
            x = np.array(x)
            labels = x.copy()

            # 将cx，cy，w，h转至左上角和右下角坐标
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            # list 2 array
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * sw, out=labels4[:, 1:])  # use with random_perspective
            # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        img4, labels4 = random_perspective(img4, labels4,
                                           degrees=0.0,
                                           translate=0.1,
                                           scale=0.5,
                                           shear=0.0,
                                           perspective=0.0,
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    def load_image(self, index):
        # laod image
        img_id = self.img_ids[index]
        img_path = self.imgid2path[img_id]
        self.coco = True if 'COCO' in img_path else False
        if self.pro_type == 'torch':
            img = Image.open(img_path)
            ori_w, ori_h = img.width, img.height
        else:
            img = cv2.imread(img_path)

        # load unnormalized annotation
        annotations = self.imgid2anns[img_id]
        gt_num = len(annotations)
        # labels shape(50, 5), 5 = [x, y, w, h, angle]
        labels = torch.zeros(self.max_labels, 5)
        categories = torch.zeros(self.max_labels, dtype=torch.int64)
        li = 0
        for ann1 in annotations:
            # if self.only_person and ann['category_id'] != 1:
            #     continue
            ann = ann1.copy()
            # ann = self._prepare(ann)
            if len(ann['bbox']) != 0:
                labels[li, 0] = ann['category_id']
                labels[li, 1:] = ann['bbox']
                categories[li] = self.catids.index(ann['category_id']+1)
                li += 1
        # if self.only_person:
        #     assert (categories == 0).all()
        gt_num = li

        # pad to square
        input_img, labels[:gt_num], pad_info = rect_to_square(img, labels[:gt_num],
                                                              self.img_size, self.pro_type, pad_value=0,
                                                              aug=self.enable_aug)

        # input_img, labels[:gt_num], pad_info = letterbox(img, labels[:gt_num], self.img_size)

        labels[:gt_num] = normalize_bbox(labels[:gt_num], self.img_size[1], self.img_size[0])
        return input_img, labels

def dilate_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素的形状和大小
    dst = cv2.dilate(binary, kernel)  # 膨胀操作
    return dst

def mask_todilate(_mask):
    _mask[_mask == 0] = 1
    _mask[_mask > 1] = 0
    _mask = dilate_demo(_mask)
    return _mask

def uniform(a, b):
    return a + np.random.rand() * (b-a)

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale 图像旋转
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear 图像剪切
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation 图像平移
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    # @ 表示叉乘 矩阵乘法
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            # 透视变换
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            # 放射变换
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy_ = xy.copy()
        xy_[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy_[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy_.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

        # targets[:, 1:5] = xy

        targets[:, [1, 3]] = targets[:, [1, 3]].clip(0, width)
        targets[:, [2, 4]] = targets[:, [2, 4]].clip(0, height)

    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=400, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return  (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])
from models import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from terminaltables import AsciiTable
import time
import Detect_0520
import argparse
from utils_junjie.parse_config import *
from utils_junjie import tools, timer, visualization
import Detect
from utils_junjie.get_module_list import *
from collections import OrderedDict
from ptflops import get_model_complexity_info
import datetime


parser = argparse.ArgumentParser(description="demo for rotated image detection")
parser.add_argument('--input_size', type=int, default=(512, 512))
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou thresshold for compute AP")

parser.add_argument('--model', type=str, default='weights_original/Oct06-23_{}.pth')
parser.add_argument('--data_config', type=str, default='config/cls.names')
parser.add_argument('--model_def', type=str, default='cfg/darknet53_original.cfg')

parser.add_argument('--preprocess_type', type=str, default='cv2', choices=['cv2', 'torch'],
                        help='image preprocess type')
args = parser.parse_args()


# ==================== Loading Model ..... ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

module_defs = parse_model_config(args.model_def)
# module_defs.pop(0)
model = RotateDetectNet(module_defs)


job_name = '{}'.format(datetime.datetime.now().strftime('%b-%d-%H'))
# iter_i = int(args.trained_model.split('.')[0].split('_')[-1])
logfile = './log_lighten_original/{}/'.format(job_name)
os.makedirs(logfile, exist_ok=True)

# ==================== prepare validation dataset =====================
val_img_dir =['/home/jie/Phd-project/RockData/VOC/val_aug/JPEGImages/']
val_json = ['/home/jie/Phd-project/RockData/VOC/val_aug/val_aug_2021.json']
class_names = parse_data_config(args.data_config)['names'].split(',')
label_index = [i + 1 for i in range(len(class_names))]
labels = dict(zip(label_index, list(class_names)))
for iter_i in range(1000, 64000, 1000):
    print(args.model.format(iter_i))
    pretrained_dict = torch.load(args.model.format(iter_i))
    model.load_state_dict(pretrained_dict['model'])

    # =================== calculate model parameters and computations ========

    flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,print_per_layer_stat=False)
    print("INFO: %s |%s" % (flops, params))

    # iter_i = pretrained_dict['iter']
    model.eval()
    model.to(device)
    # ===================== do validation ==============================
    # model_eval = Detect_0520.Detector(model=model, iteration=iter_i, conf_thres=args.conf_thres, nms_thres=args.nms_thres,
    #                                              iou_thres=args.iou_thres, input_size=args.input_size, class_name=class_names)
    # metrics_output = model_eval.forward(val_img_dir, gt_path=val_json, input_size=args.input_size, class_name=class_names,
    #                          pro_type=args.preprocess_type, log_file = logfile)

    model_eval = Detect_0520.Detector(model=model, label=labels, iteration=iter_i, conf_thres=args.conf_thres,
                                      nms_thres=args.nms_thres,
                                      iou_thres=args.iou_thres, input_size=args.input_size, class_name=class_names)
    metrics_output = model_eval.forward(val_img_dir, gt_path=val_json, input_size=args.input_size, log_file=logfile,
                                        class_name=class_names, pro_type=args.preprocess_type)

# # ===================== do validation ==============================
# model_eval = Detect.Detector(model=model, conf_thres=args.conf_thres, nms_thres=args.nms_thres,
#                                              iou_thres=args.iou_thres, input_size=args.input_size, class_name=class_names)
# metrics_output = model_eval.forward(val_img_dir, gt_path=val_json, input_size=args.input_size,
#                          pro_type=args.preprocess_type)
#
# if metrics_output is not None:
#     precision, recall, AP, ap_class, iou_thres = metrics_output
#     evaluation_metrics = [
#         ("validation/precision", precision.mean()),
#         ("validation/recall", recall.mean()),
#         ("validation/mAP", AP.mean()),
#     ]
#
#     # Print class APs and mAP
#     ap_table = [["Index", "Class name", "Precision(0.5)", "Recall(0.5)", "AP(0.5)", "AP(0.75)", "AP(0.5:0.95)"]]
#     for i, c in enumerate(ap_class):
#         ap_table += [[c, class_names[c], "%.5f"%precision[c][0], "%.5f"%recall[0][i], "%.5f" % AP[c][0],
#                       "%.5f" % AP[c][iou_thres.index(0.75)], "%.5f" % AP[c].mean()]]
#     print(AsciiTable(ap_table).table)
#
#     # with open('./log.txt', 'a') as fd:
#     #     fd.write('Iteration: %d\n'%iter_i + AsciiTable(ap_table).table +'\n')


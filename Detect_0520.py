import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms.functional as tvf

from utils_junjie import dataloader, utils, voc_ap
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os
import json
import shutil
from terminaltables import AsciiTable


class Detector():
    '''
    Wrapper of image object detectors.

    Args:
        model_name: str, currently only support 'rapid'
        weights_path: str, path to the pre-trained network weights
        model: torch.nn.Module, used only during training
        conf_thres: float, confidence threshold
        input_size: int, input resolution
    '''

    def __init__(self, model=None, **kwargs):
        assert torch.cuda.is_available()
        if model:
            self.model = model
            self.iter_i = kwargs.get('iteration', None)
            self.conf_thres = kwargs.get('conf_thres', None)
            self.nms_thres = kwargs.get('nms_thres', None)
            self.class_name = kwargs.get('class_name', None)
            self.labels = kwargs.get('label', None)
            self.input_size = kwargs.get('input_size', None)

            self.iou_thres = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
            # self.iou_thres = (0.5,)
            self.rec_thres = torch.linspace(0, 1, steps=101)

    def forward(self, img_dir, **kwargs):
        '''
        Run on a sequence of images in a folder.

        Args:
            img_dir: str
            input_size: int, input resolution
            conf_thres: float, confidence threshold
        '''
        gt_path = kwargs['gt_path'] if 'gt_path' in kwargs else None

        ims = dataloader.Images4Detector(img_dir, gt_path, self.labels, **kwargs)  # TODO
        dts = self._detect_iter(iter(ims), ims, **kwargs)
        return dts

    def _detect_iter(self, iterator, ims, **kwargs):
        # =============== step 1 处理ground truth信息 ======================
        cls = kwargs['class_name']
        cls = dict(zip(map(str, range(len(cls))), cls))
        results_files_path = kwargs['log_file']
        TEMP_FILES_PATH = ".temp_files"
        os.makedirs(TEMP_FILES_PATH, exist_ok=True)
        # results_files_path = "results"
        # if os.path.exists(results_files_path):  # if it exist already
        #     # reset the results directory
        #     shutil.rmtree(results_files_path)
        # os.makedirs(results_files_path)
        # # if draw_plot:
        # os.makedirs(os.path.join(results_files_path, "classes"))
        # 统计gt中每个类目标框的数量以及每个类对应有多少张图片
        gt_counter_per_class = {}
        counter_images_per_class = {}
        for _, key in enumerate(ims.imgid2anns.keys()):
            bounding_boxes = []
            already_seen_classes = []
            for ann in ims.imgid2anns[key]:
                box = utils.xywh2xyxy(ann['bbox'])
                class_name = cls[str(ann['category_id'])]
                bbox = str(float(box[0])) + " " + str(float(box[1])) + " " + str(float(box[2])) + " " + str(float(box[3]))

                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)
            # 重新生成并保存json格式的gt信息
            with open(TEMP_FILES_PATH + "/" + ann['image_id'] + "_ground_truth.json", 'w') as outfile:
                json.dump(bounding_boxes, outfile)
        # =============== step 1 处理ground truth信息 ======================

        # =============== step 2 模型检测 ======================
        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        # gt_classes = ['shale', 'monzogranite']
        # for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = [[] for _ in range(len(gt_classes))]
        for _ in tqdm(range(len(iterator))):
            pil_frame, anns, img_id = next(iterator)
            detections = self._predict_pil(pil_img=pil_frame, **kwargs)
            for line in detections:
                tmp_class_name = cls[str(int(line.data.numpy()[-1]))]
                left = float(line.data.numpy()[0])
                top = float(line.data.numpy()[1])
                right = float(line.data.numpy()[2])
                bottom = float(line.data.numpy()[3])
                confidence = float(line.data.numpy()[4])
                bbox = str(left) + " " + str(top) + " " + str(right) + " " + str(bottom)
                if tmp_class_name in gt_classes:
                    # 将模型检测的每个框，分别按类别，和对应的imageid保存json格式
                    bounding_boxes[gt_classes.index(tmp_class_name)].append({"confidence": confidence, "file_id": img_id, "bbox": bbox})
        # sort detection-results by decreasing confidence
        for index in range(len(gt_classes)):
            bounding_boxes[index].sort(key=lambda x: float(x['confidence']), reverse=True)
            with open(TEMP_FILES_PATH + "/" + gt_classes[index] + "_dr.json", 'w') as outfile:
                json.dump(bounding_boxes[index], outfile)


        """
         Calculate the AP for each class
        """
        Precision_iou_thres = np.zeros((len(gt_classes), len(self.iou_thres)))
        Recall_iou_thres = np.zeros((len(gt_classes), len(self.iou_thres)))
        AP_iou_thres = np.zeros((len(gt_classes), len(self.iou_thres)))
        with open(results_files_path + "results.txt", 'a+') as results_file:
            results_file.write("# AP and precision/recall per class\n")
            for iou_index, MINOVERLAP in tqdm(enumerate(self.iou_thres), desc='AP for IOU(0.5:095:0.05)'):
                sum_AP = 0.0
                # ap_dictionary = {}
                # lamr_dictionary = {}
                # open file to store the results
                count_true_positives = {}
                for class_index, class_name in enumerate(gt_classes):
                    count_true_positives[class_name] = 0
                    """
                     Load detection-results of that class
                    """
                    dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
                    dr_data = json.load(open(dr_file))

                    """
                     Assign detection-results to ground-truth objects
                    """
                    nd = len(dr_data)
                    tp = [0] * nd  # creates an array of zeros of size nd
                    fp = [0] * nd
                    conf = [float(d['confidence']) for d in dr_data]
                    for idx, detection in enumerate(dr_data):
                        file_id = detection["file_id"]
                        # assign detection-results to ground truth object if any
                        # open ground-truth with that file_id
                        gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                        ground_truth_data = json.load(open(gt_file))
                        ovmax = -1
                        gt_match = -1
                        # load detected object bounding-box
                        bb = [float(x) for x in detection["bbox"].split()]
                        for obj in ground_truth_data:
                            # look for a class_name match
                            if obj["class_name"] == class_name:
                                bbgt = [float(x) for x in obj["bbox"].split()]
                                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                                iw = bi[2] - bi[0] + 1
                                ih = bi[3] - bi[1] + 1
                                if iw > 0 and ih > 0:
                                    # compute overlap (IoU) = area of intersection / area of union
                                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]+ 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                    ov = iw * ih / ua
                                    if ov > ovmax:
                                        ovmax = ov
                                        gt_match = obj

                        # assign detection as true positive/don't care/false positive
                        # set minimum overlap
                        # MINOVERLAP = 0.5
                        min_overlap = MINOVERLAP

                        if ovmax >= min_overlap:
                            if "difficult" not in gt_match:
                                if not bool(gt_match["used"]):
                                    # true positive
                                    tp[idx] = 1
                                    gt_match["used"] = True
                                    count_true_positives[class_name] += 1
                                    # update the ".json" file
                                    # with open(gt_file, 'w') as f:
                                    #     f.write(json.dumps(ground_truth_data))
                                else:
                                    # false positive (multiple detection)
                                    fp[idx] = 1
                        else:
                            # false positive
                            fp[idx] = 1
                            if ovmax > 0:
                                status = "INSUFFICIENT OVERLAP"

                    # print(tp)
                    # compute precision/recall
                    cumsum = 0
                    for idx, val in enumerate(fp):
                        fp[idx] += cumsum
                        cumsum += val
                    cumsum = 0
                    for idx, val in enumerate(tp):
                        tp[idx] += cumsum
                        cumsum += val
                    # print(tp)
                    rec = tp[:]
                    for idx, val in enumerate(tp):
                        rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                    # print(rec)
                    prec = tp[:]
                    for idx, val in enumerate(tp):
                        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                    # print(prec)

                    ap, mrec, mprec = voc_ap.voc_ap(rec[:], prec[:])
                    sum_AP += ap
                    text = "{0:.2f}%".format(
                        ap * 100) + " = " + class_name + " AP(%d) "%(MINOVERLAP*100)  # class_name + " AP = {0:.2f}%".format(ap*100)
                    text_re = "{0:.2f}%".format(
                        rec[-1] if len(rec)>0 else 0 * 100) + " = " + class_name + " Recall(%d) "%(MINOVERLAP*100)  # class_name + " AP = {0:.2f}%".format(ap*100)
                    text_pre = "{0:.2f}%".format(
                        prec[-1] if len(prec)>0 else 0* 100) + " = " + class_name + " Precision(%d) "%(MINOVERLAP*100)  # class_name + " AP = {0:.2f}%".format(ap*100)
                    print(text + '\n' + text_re + '\n' + text_pre)
                    Precision_iou_thres[class_index, iou_index] = prec[-1] if len(prec)>0 else 0
                    Recall_iou_thres[class_index, iou_index] = rec[-1] if len(rec)>0 else 0
                    AP_iou_thres[class_index, iou_index] = ap

                    """
                     Draw plot
                    """
                    draw_plot = True
                    if draw_plot:
                        plt.plot(rec, prec, '-', linewidth=0.5, color='red', label='pr curve')
                        plt.plot(rec, conf, '-', linewidth=0.5, color='green', label='conf')
                        plt.legend(loc="lower right", fontsize=10, bbox_to_anchor=(1, 1),
                                   ncol=1, columnspacing=0.1,
                                   labelspacing=0.2, markerscale=1, shadow=True, borderpad=0.2, handletextpad=0.2)

                        plt.grid(c='b', linewidth=0.25)
                        # add a new penultimate point to the list (mrec[-2], 0.0)
                        # since the last line segment (and respective area) do not affect the AP value
                        area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                        area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                        plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                        # # set window title
                        # fig = plt.gcf()  # gcf - get current figure
                        # fig.canvas.set_window_title('AP ' + class_name)
                        # set plot title
                        plt.title('class: ' + text)
                        # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                        # set axis titles
                        plt.xlabel('Recall')
                        plt.ylabel('Precision')
                        # optional - set axes
                        axes = plt.gca()  # gca - get current axes
                        axes.set_xlim([0.0, 1.0])
                        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space

                        # save the plot
                        plt.savefig(results_files_path + class_name + "_iou(%.2f).png"%(MINOVERLAP))
                        plt.cla()  # clear axes for next plot



            # # Print class APs and mAP
            ap_table = [
                ["Index", "Class name", "Precision(0.5)", "Recall(0.5)", "AP(0.5)", "AP(0.75)", "AP(0.5:0.95)"]]
            for c in range(len(gt_classes)):
                ap_table += [[c, gt_classes[c], "%.5f" % Precision_iou_thres[c][0], "%.5f" % Recall_iou_thres[c][0], "%.5f" % AP_iou_thres[c][0],
                              "%.5f" % AP_iou_thres[c][self.iou_thres.index(0.75)], "%.5f" % AP_iou_thres[c].mean()]]
            print(AsciiTable(ap_table).table)

            results_file.write('Iteration: %d\n'%self.iter_i + AsciiTable(ap_table).table +'\n')

        return Precision_iou_thres, Recall_iou_thres, AP_iou_thres, gt_classes, self.iou_thres

    def _predict_pil(self, pil_img, **kwargs):
        '''
        Args:
            pil_img: PIL.Image.Image
            input_size: int, input resolution
            conf_thres: float, confidence threshold
        '''
        input_size = kwargs.get('input_size', self.input_size)
        conf_thres = kwargs.get('conf_thres', self.conf_thres)
        nms_thres = kwargs.get('nms_thres', self.nms_thres)
        pro_type = kwargs['pro_type'] if 'pro_type' in kwargs else None
        # assert isinstance(pil_img, Image.Image), 'input must be a PIL.Image'
        assert input_size is not None, 'Please specify the input resolution'
        assert conf_thres is not None, 'Please specify the confidence threshold'

        # pad to square
        input_img, _, pad_info = utils.rect_to_square(pil_img, None, input_size, pro_type, 0)
        if pro_type=='torch':
            input_ori = tvf.to_tensor(input_img)
            input_ = input_ori.unsqueeze(0)
        else:
            input_ori = torch.from_numpy(input_img).permute((2, 0, 1)).float() / 255.
            input_ = input_ori.unsqueeze(0)

        assert input_.dim() == 4
        input_ = input_.cuda()
        with torch.no_grad():
            dts = self.model(input_).cpu()

        dts = dts.squeeze()
        # post-processing
        dts = dts[dts[:, 4:].max(-1)[0] >= conf_thres]
        if len(dts):
            dts = utils.non_max_suppression(dts, conf_thres, nms_thres)
            dts = utils.detection2original(dts, pad_info.squeeze())
        return dts


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse
from utils_junjie.parse_config import *
from utils_junjie.utils import *
from utils_junjie.visualization import *
from utils_junjie.get_module_list import *
from PIL import Image, ImageDraw, ImageFont

import cv2


parser = argparse.ArgumentParser(description="demo for rotated image detection")
parser.add_argument('--model_name', type=str, default='rapid',
                    help='the name of model')
parser.add_argument('--model_def', type=str, default='cfg/prune_0.9_prune_rotate_detection.cfg')
parser.add_argument('--model', type=str, default='weights_pruned_aug/Oct05-16_63000.pth',
                    help='model path')
parser.add_argument('--img_path', type=str, default='/home/jie/Phd-project/RockData/VOC/val_aug/JPEGImages',
                    help='image path')
parser.add_argument('--result_path', dest='result_path', help='directory to load images for demo',
                      default='/home/jie/Phd-project/RockData/VOC/val_aug/pruned-aug-result')
parser.add_argument('--input_size', type=int, default=(512, 512))
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument('--visualize', type=bool, default=True)
parser.add_argument('--preprocess_type', type=str, default='cv2', choices=['cv2', 'torch'],
                        help='image preprocess type')
args = parser.parse_args()


# # %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


module_defs = parse_model_config(args.model_def)
model = RotateDetectNet(module_defs)

pretrained_dict = torch.load(args.model)
model.load_state_dict(pretrained_dict['model'])

# result_path = '/media/jie/Work/Public_Datasets/pano/20210117/empty_img'
# os.makedirs(result_path, exist_ok=True)
os.makedirs(args.result_path, exist_ok=True)
model.to(device)
time_cost = 0
classes = ['monzogranite', 'shale', 'volcanic_tuff', 'sandstone']

with torch.no_grad():
    for root, folder, files in os.walk(args.img_path):
        if len(files)>0:
            count_num=0
            print("INFO: " + root)
            for name in files:
                if not name.endswith('.jpg'):
                    continue
                count_num+=1
                print('INFO: %d/%d'%(count_num, len(files)))
                pro_type = args.preprocess_type
                start = time.time()
                if pro_type=='torch':
                    pil_img = Image.open(os.path.join(args.img_path + str(i), name))
                else:
                    pil_img = cv2.imread(os.path.join(root, name))

                print("INFO: Input Preprocess time {}".format(time.time() - start))
                input_img, _, pad_info = rect_to_square(pil_img, None, args.input_size, pro_type, 0)

                input_ori = torch.from_numpy(input_img).permute((2, 0, 1)).float() / 255.
                input_ = input_ori.unsqueeze(0)

                assert input_.dim() == 4
                input_ = input_.cuda()
                start = time.time()
                dts = model(input_).cpu()
                time_cost += time.time() -start
                print("INFO: detection forward time {}".format(time.time() - start))
                np_img = np.array(pil_img)
                dts = dts.squeeze()
                # post-processing
                dts = dts[dts[:, 4:].max(-1)[0] >= args.conf_thres]
                if len(dts):
                    # _, idx = torch.topk(dts[:, 4], k=1000)
                    # dts = dts[idx, :]
                    detections = non_max_suppression(dts, args.conf_thres, args.nms_thres)
                    # detections = nms(dts, is_degree=True, nms_thres=0.3, img_size=args.input_size)
                    detections = detection2original(detections, pad_info.squeeze())
                    for bb in detections:
                        # np_img = cv2ImgAddText(np_img, classes[int(bb[-1])] + ': %.2f'%(bb[-2]),
                        #                        int(bb[0]), int(bb[1]), (255, 255, 255))
                        cv2.putText(np_img, classes[int(bb[-1])] + ':%.2f'%(bb[-2]), (int(bb[0]), int(bb[1])+20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0,  0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(np_img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color=(255, 0, 0), thickness=1,
                                      lineType=cv2.LINE_4)
                # np_result = cv2.resize(np_img, (640, 360))
                # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                # cv2.imshow('result', np_img)
                # cv2.waitKey()
                cv2.imwrite(os.path.join(args.result_path, name), np_img)
            # print("INFO: Average Inference time is {}".format(time_cost / len(files)))
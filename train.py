# This is the main training file we are using
import os
import argparse
import random

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

from datasets_crop import Dataset4YoloAngle
import Detect_0520
from utils_junjie.parse_config import *
from utils_junjie import tools, timer, logger
from utils_junjie.utils import *
from utils_junjie.get_module_list import *
from collections import OrderedDict
import datetime
import time
from terminaltables import AsciiTable
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=(512, 512))
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou thresshold for compute AP")

    parser.add_argument('--checkpoint', type=str, default='Sep16-15_17000.pth')
    # parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_config', type=str, default='config/cls.names')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.finetune.yaml', help='hyperparameters path')
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--img_interval', type=int, default=500)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--model_def', type=str, default='cfg/RDNet.cfg')

    parser.add_argument('--debug', action='store_true')  # default=True)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=2,
                        help='number of the gpus')
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument('--data_pipeline', type=str, default='no_dali', choices=['dali', 'no_dali'],
                        help='data preprocessing pipline to use')
    parser.add_argument('--preprocess_type', type=str, default='cv2', choices=['cv2', 'torch'],
                        help='image preprocess type')

    parser.add_argument('--visulization', type=bool, default=False,
                        help='image preprocess type')
    parser.add_argument('--enable_aug', type=bool, default=True,
                        help='image enable_aug type')
    parser.add_argument('--enable_mosaic', type=bool, default=True,
                        help='image enable_aug type')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--add_mask', '-mask', type=bool, default=False,
                        help='add truck mask to bus dataset')
    parser.add_argument('--s', type=float, default=0.01, help='scale sparse rate')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.cuda and torch.cuda.is_available()  # Currently do not support CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -------------------------- settings ---------------------------
    # assert not args.adversarial
    job_name = '{}'.format(datetime.datetime.now().strftime('%b%d-%H'))
    logfile = './log/{}/'.format(job_name)
    os.makedirs(logfile, exist_ok=True)
    # dataloader setting
    batch_size = args.batch_size
    num_cpu = 0 if batch_size == 1 else 4
    subdivision = 128 // batch_size

    # SGD optimizer
    decay_SGD = 0.0005 * batch_size * subdivision
    print('effective batch size = {} * {}'.format(batch_size, subdivision))
    # dataset setting
    print('initialing dataloader...')

    train_img_dir = ['*/train/JPEGImages']
    train_json = ['*/train_2021.json']


    val_img_dir =['*/val/JPEGImages']
    val_json = ['*/val_2021.json']
    class_names = parse_data_config(args.data_config)['names'].split(',')
    with open(args.hyp) as f:
        hyp = yaml.load(f)  # load hyps
        # hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
    # lr_SGD = 0.0001 / batch_size / subdivision
    lr_SGD = 0.0001


    # Learning rate setup
    def burnin_schedule(i):
        burn_in = 500
        if i < burn_in:
            factor = (i / burn_in) ** 2
        elif i < 10000:
            factor = 1.0
        elif i < 20000:
            factor = 0.3
        else:
            factor = 0.1
        return factor


    label_index = [i + 1 for i in range(len(class_names))]
    labels = dict(zip(label_index, list(class_names)))

    dataset = Dataset4YoloAngle(train_img_dir, train_json, labels, args.input_size, args.enable_aug, args.enable_aug,
                                pro_type=args.preprocess_type, hpy=hyp)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=False)
    dataiterator = iter(dataloader)

    module_defs = parse_model_config(args.model_def)
    # module_defs.pop(0)
    model = RotateDetectNet(module_defs)
    model.to(device)

    # # =================对初始化模型进行赋值操作================
    # model_dict = model.state_dict()
    #
    # start_iter = -1
    # if args.checkpoint:
    #     print("loading ckpt...", args.checkpoint)
    #     weights_path = os.path.join('./weights/', args.checkpoint)
    #     pretrained_dict = torch.load(weights_path)['model']
    #
    #
    #     new_state_dict = OrderedDict()
    #     for k, v in pretrained_dict.items():
    #         if k.split('.')[1] not in ['81', '93', '105']:
    #             new_state_dict[k] = v
    #     model_dict.update(new_state_dict)
    #     model.load_state_dict(model_dict)
    #     model.to(device)
    # # # ==================

    start_iter = -1
    if args.checkpoint:
        print("loading ckpt...", args.checkpoint)
        weights_path = os.path.join('./weights/', args.checkpoint)
        state = torch.load(weights_path)

        model.load_state_dict(state['model'])
        start_iter = state['iter']
    # logger = logger.Logger('./log_lighten_aug/{}'.format(job_name))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_SGD, momentum=0.9, dampening=0,
                                weight_decay=decay_SGD)

    # optimizer.load_state_dict(state['optimizer'])
    print('begin from iteration: {}'.format(start_iter))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    scheduler.last_epoch = start_iter - 1
    print("INFO: %d" % start_iter)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule, last_epoch=-1)

    # start training loop
    best_value = 0
    start_time = timer.tic()
    # start_iter = state['iter']
    for iter_i in range(start_iter, 800000):
        model.train()
        # # evaluation
        if iter_i % args.eval_interval == 0 and (iter_i >=100):
            with timer.contexttimer() as t0:
                model.eval()

                model_eval = Detect_0520.Detector(model=model, label=labels, iteration=iter_i, conf_thres=args.conf_thres, nms_thres=args.nms_thres,
                                             iou_thres=args.iou_thres, input_size=args.input_size, class_name=class_names)
                metrics_output = model_eval.forward(val_img_dir, gt_path=val_json, input_size=args.input_size, log_file = logfile,
                                                    class_name=class_names, pro_type=args.preprocess_type)

            model.train()

        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            start = time.time()
            try:
                imgs, targets = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets = next(dataiterator)  # load a batch
            if args.visulization:
                for index, img in enumerate(imgs):
                    target = targets[index]
                    np_img = (img.data.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                    boxes = target.numpy()
                    for box in boxes:
                        if sum(box) == 0:
                            continue
                        x = box[1] * np_img.shape[1]
                        y = box[2] * np_img.shape[0]
                        w = box[3] * np_img.shape[1]
                        h = box[4] * np_img.shape[0]
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        cv2.rectangle(np_img, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 255), thickness=1, lineType=cv2.LINE_4)
                    cv2.namedWindow('1', cv2.WINDOW_NORMAL)
                    cv2.imshow('1', np_img)
                    cv2.waitKey()

            # visualization.imshow_tensor(imgs)
            imgs = imgs.cuda() if args.cuda else imgs
            torch.cuda.reset_max_memory_allocated()
            loss = model(imgs, targets)

            loss.backward()
            #print("INFO: Each Iteration time is {}".format(time.time() - start))
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % args.print_interval == 0:
            sec_used = timer.tic() - start_time
            time_used = timer.sec2str(sec_used)
            avg_iter = timer.sec2str(sec_used / (iter_i + 1 - start_iter))
            avg_epoch = avg_iter / batch_size / subdivision * 118287
            print('\nTotal time: {}, iter: {}, epoch: {}'.format(time_used, avg_iter, avg_epoch))
            # current_lr = scheduler.get_lr()[0] * batch_size * subdivision
            current_lr = scheduler.get_lr()[0]
            print('[Iteration {}] [learning rate {}]'.format(iter_i, '%.5f'% current_lr),
                  '[Total loss {}] [img size {}]'.format('%.2f'% loss, dataset.img_size))
            print(model.loss_str)
            with open(logfile + 'loss.txt', 'a+') as fd:
                fd.write(model.all_loss +'\n')
            max_cuda = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            print('Max GPU memory usage: {} GigaBytes'.format(max_cuda))
            torch.cuda.reset_max_memory_allocated(0)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', '{}_{}.pth'.format(job_name, iter_i))
            torch.save(state_dict, save_path)


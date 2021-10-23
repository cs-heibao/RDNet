import torch.nn as nn
from models.rapid_modify import *
import torch

class RotateDetectNet(nn.Module):
    def __init__(self, module_defs, **kwargs):
        super(RotateDetectNet, self).__init__()
        # self.model = model
        self.module_defs = module_defs
        self.module_list = create_modules(self.module_defs, **kwargs)


    def forward(self, x, labels=None, **kwargs):
        '''
                x: a batch of images, e.g. shape(8,3,608,608)
                labels: a batch of ground truth
                '''
        assert x.dim() == 4
        loss = 0
        self.loss_str = ''
        self.all_loss = ''
        self.img_size = x.shape
        layer_outputs, yolo_outputs, onnx_export = [], [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # go through backbone
            if module_def['type'] in ['convolutional', 'upsample']:
                x = module(x)
            elif module_def['type'] =='route':
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def['type'] =='shortcut':
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] =='yolo':
                onnx_export.append(x)
                x, layer_loss = module[0](x, self.img_size, labels)  # module是nn.Sequential()，所以要取[0]
                yolo_outputs.append(x)
                if labels is not None:
                    self.loss_str = self.loss_str + module[0].loss_str + '\n'
                    self.all_loss = self.all_loss + '%.3f'%module[0].loss_xy+' '+'%.3f'%module[0].loss_wh+' '+'%.3f'%module[0].loss_obj+' '
                    loss += layer_loss

            layer_outputs.append(x)
        # return onnx_export

        if labels is None:
            # # assert boxes_L.dim() == 3
            boxes = torch.cat(yolo_outputs, dim=1)
            return boxes
        else:
            # # check all the gt objects are assigned
            # self.loss_str = self.pred_L.loss_str + '\n' + self.pred_M.loss_str + \
            #                 '\n' + self.pred_S.loss_str
            return loss

def create_modules(module_defs, **kwargs):
    output_filters = [3]
    module_list = nn.ModuleList()

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2

            modules.add_module("{}".format(0),
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                    padding_mode='zeros',
                )
            )

            if bn:
                modules.add_module("{}".format(1), nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("{}".format(2), nn.LeakyReLU(0.1, inplace=True))
            if module_def["activation"] == "relu":
                modules.add_module("{}".format(2), nn.ReLU())


        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                # modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module("_debug_padding_{}".format(module_i), nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            # modules.add_module(f"maxpool_{module_i}", maxpool)
            modules.add_module("maxpool_{}".format(module_i), maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"].split(',')[0]), mode="nearest")
            # upsample = Upsample(scale_factor=(int(module_def['stride'].split(',')[1]),int(module_def['stride'].split(',')[2])), mode="nearest")
            modules.add_module("upsample_{}".format(module_i), upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            # modules.add_module(f"route_{module_i}", EmptyLayer())
            modules.add_module("route_{}".format(module_i), EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            # if "ECA" in module_def.keys():
            #     eca = ECA_layer(k_size=3)
            #     modules.add_module("eca_{}".format(module_i), eca)
            modules.add_module("shortcut_{}".format(module_i), EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [float(x) for x in module_def["anchors"].split(",")]
            anchors = [[anchors[i], anchors[i + 1]] for i in range(0, len(anchors), 2)]
            num_classes = int(module_def["classes"])
            anchors_all = torch.Tensor(anchors).float()
            index = torch.Tensor(anchor_idxs).long()

            # Define detection layer
            yolo_layer = YOLOBranch(anchors_all, index, num_classes, **kwargs)
            # modules.add_module(f"yolo_{module_i}", yolo_layer)
            modules.add_module("yolo_{}".format(module_i), yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)  # filter保存了输出的维度

    return module_list
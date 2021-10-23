from math import pi
import torch

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    """
    assert bboxes_a.dim() == bboxes_b.dim() == 2
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
    
def xywha2vertex(box, is_degree, stack=True):
    '''
    Args:
        box: tensor, shape(batch,4), 4=(x,y,w,h), xy is center,

    Return:
        tensor, shape(batch,4,2): topleft, topright, br, bl
    '''
    assert is_degree == False and box.dim() == 2 and box.shape[1] >= 4
    batch = box.shape[0]
    device = box.device

    center = box[:,0:2]
    w = box[:,2]
    h = box[:,3]

    # calculate two vector
    verti = torch.empty((batch,2), dtype=torch.float32, device=device)
    verti[:,0] = (h/2)
    verti[:,1] = - (h/2)

    hori = torch.empty(batch,2, dtype=torch.float32, device=device)
    hori[:,0] = (w/2)
    hori[:,1] = (w/2)


    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    if not stack:
        return torch.cat([tl,tr,br,bl], dim=1)
    return torch.stack((tl,tr,br,bl), dim=1)

from pycocotools import mask as maskUtils
def iou_rle(boxes1, boxes2, xywha, is_degree=True, **kwargs):
    r'''
    use mask method to calculate IOU between boxes1 and boxes2

    Arguments:
        boxes1: tensor or numpy, shape(N,5), 5=(x, y, w, h, angle 0~90)
        boxes2: tensor or numpy, shape(M,5), 5=(x, y, w, h, angle 0~90)
        xywha: True if xywha, False if xyxya
        is_degree: True if degree, False if radian

    Return:
        iou_matrix: tensor, shape(N,M), float32, 
                    ious of all possible pairs between boxes1 and boxes2
    '''
    assert xywha == True and is_degree == True

    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        print('Warning: bounding boxes are np.array. converting to torch.tensor')
        # convert to tensor, (batch, (x,y,w,h,a))
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    assert boxes1.device == boxes2.device
    device = boxes1.device
    boxes1, boxes2 = boxes1.cpu().clone().detach(), boxes2.cpu().clone().detach()
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    assert boxes1.shape[1] == boxes2.shape[1] == 4

    size = kwargs.get('img_size', 2048)
    # if isinstance(size, tuple):
    if isinstance(size, int):
        h, w = size, size
    elif len(size) == 2:
        h, w = size[0], size[1]
    else:
        h, w = size[2], size[3]
    if 'normalized' in kwargs and kwargs['normalized'] == True:
        # the [x,y,w,h] are between 0~1
        # assert (boxes1[:,:4] <= 1).all() and (boxes2[:,:4] <= 1).all()
        boxes1[:,0] *= w
        boxes1[:,1] *= h
        boxes1[:,2] *= w
        boxes1[:,3] *= h
        boxes2[:,0] *= w
        boxes2[:,1] *= h
        boxes2[:,2] *= w
        boxes2[:,3] *= h

    b1 = xywha2vertex(boxes1, is_degree=False, stack=False).tolist()
    b2 = xywha2vertex(boxes2, is_degree=False, stack=False).tolist()
    debug = 1
    
    b1 = maskUtils.frPyObjects(b1, h, w)
    b2 = maskUtils.frPyObjects(b2, h, w)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    return torch.from_numpy(ious).to(device=device)

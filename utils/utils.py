from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


"""
输入：pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
pred_boxes: yolo层中预测出来的所有方框的坐标x,y,w,h，维度为2x3x13x13x4 = Batch x numberOfAnchors x 格子数量 x 格子数量 x 方框坐标属性个数
pred_conf: yolo层中预测出来的所有方框含有目标的概率得分，维度为2x3x13x13
pred_cls: yolo层中预测出来的所有方框所含目标的为各个类的概率得分，维度为2x3x13x13x80
target: 维度为2x50x5 = batch x 一张图最大目标数量 x 方框属性数量(类别,x,y,w,h)
anchors: 每个cell的对应的缩小后的anchor尺寸，维度为3x2，每行对应一个anchor尺寸
num_anchors：每个cell对应的anchor的数量3
num_classes：需要检测的物体类别，coco为80
grid_size：传入yolo层特征图的宽高大小13
ignore_thres：阈值0.5
img_dim：输入网络的图片的尺寸416

输出：nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls
nGT: 真实标签方框个数
nCorrect：正确检测出来的个数
mask:
conf_mask:
tx,ty: 真实方框与anchor的中心坐标的偏移量
tw,th：真实方框与anchor的宽高缩放比值的对数值
tconf：
tcls：
"""


def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    # nB为一个batch含有的图片数量
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    # 生成一个维度为2x3x13x13元素为0的tensor，mask用于后面来标记真正负责预测出目标的anchor，将它设置为1
    mask = torch.zeros(nB, nA, nG, nG)
    # 生成一个维度为2x3x13x13的元素为1的tensor，conf_mask用于标记含有目标得分的tensor中对应的所有anchors中，一个anchor是否真正
    # 负责一个目标检测。当anchor负责一个目标检测时，将conf_mask对应元素设置为1
    # ??????????????????
    # conf_mask还没有明白，后面为什么要先对与真实方框IOU大于0.5的anchor的conf_mask位置标记为0，然后又将IOU最大的那个位置标记为1？
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    # 生成一个维度为2x3x13x13的元素为0的tensor，当一个anchor真正负责检测一个目标时，令tconf中对应这个anchor位置的元素为1，表示这
    # 里有一个目标
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    # 用于记录每个anchor的对应目标的类别，初始设置为0
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    # 用于记录真实标签的个数
    nGT = 0
    # 用于记录检测正确的个数
    nCorrect = 0
    # 遍历每一张图片
    for b in range(nB):
        # 遍历每张图片的所有标签，这里target给定的标签为target.shape[1]=50个，只有一部分是真正的标签，其余为0
        for t in range(target.shape[1]):
            # target[b, t]，一个batch中第b张图片的第t个标签,是一个长度为5的tensor。如果这个标签的所有值都是0，则不是真正的标签，跳过
            if target[b, t].sum() == 0:
                continue
            # 当标签是真的时候，nGT加1
            nGT += 1
            # Convert to position relative to box，将真实标签中的x,y,w,h转换为相对于输入yolo层特征图的坐标，这里nG=13。gx,gy是
            # 相对于特征图的左上角坐标，表示的是真实方框的中心坐标。gw,gh为真实标签方框相对于输入yolo层特征图的大小
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices，对gx,gy向下取整，得到标签在13x13个格子中所在格子的左上角坐标
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box，得到时标签方框的宽高tensor，变成一个二维tensor，维度为1x4，前两个元素为0，后两个元素为方框宽高
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box，anchor_shapes维度为3x4，前两列元素为0，后两列每个格子对应的3个anchors的宽高
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes，计算这个真实标签与它所在格子对应的3个anchors的各自的IOU。anch_ious
            # 是长度为3的tensor
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            # conf_mask[b, anch_ious > ignore_thres, gj, gi]是提取出conf_mask tensor中真实标签所在格子对应的3个anchors与真实
            # 方框的IOU值大于0.5的元素，然后令它为0
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box，得到与真实方框IOU最大的anchor的序号，即这个anchor用于负责检测这个真实标签目标
            best_n = np.argmax(anch_ious)
            # Get ground truth box，真实标签方框属性tensor，维度为1x4，此时的gx,gy,gw,gh都是相对于特征图的坐标
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction，best_n是真实方框所在格子对应的3个anchors之中，与真实方框的IOU最大的anchor的序号
            # 这里就将由这个anchor预测出来的方框属性值(x,y,w,h)作为这个目标最好的预测值,pred_box的维度为1x4
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            # 将最好的预测值的anchor位置标记为1，即真正负责预测一个目标的anchor
            mask[b, best_n, gj, gi] = 1
            # 将负责检测这个目标的anchor含有目标的标记符号标记为1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            # tx[b, best_n, gj, gi]，ty[b, best_n, gj, gi]表示真正负责检测这个标签方框目标的anchor所对应的真实标签与这个标签所
            # 在格子左上角在x,y方向上的偏移量
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            # tw[b, best_n, gj, gi],th[b, best_n, gj, gi]表示真正负责检测这个标签方框目标的anchor所对应的真实标签与这个anchor
            # 的宽高比值的对数值。对应yolov3论文中2.1节中tw,th的计算，将tw,th用对数形式表达即可。此时计算的是真实标签与anchor尺寸比值
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label,得到目标类别序号
            target_label = int(target[b, t, 0])
            # 将负责预测的anchor对应的目标类别的标记符号标记为1
            tcls[b, best_n, gj, gi, target_label] = 1
            # 令tconf中负责检测一个目标的anchor对应位置的元素为1，表明这个anchor对应一个真实标签
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            # 计算真实标签与由负责预测这个目标的anchor预测出来的方框的IOU值
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            # 在负责预测这个目标的anchor对应的各个类别概率得分中，找到最大的得分那个类的序号，作为预测类别
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            # 得到负责检测这个目标的anchor含有目标的得分
            score = pred_conf[b, best_n, gj, gi]
            # 当真实标签与由负责预测这个目标的anchor预测出来的方框的IOU值大于0.5，预测类别与真实类别相同，这个anchor含有目标的得分
            # 大于0.5时，真正检测到了一个目标，nCorrect加1
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])

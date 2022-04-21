from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def bbox_iou(anchors: Tensor, targets: Tensor) -> Tensor:
    num_batch, num_objects = targets.shape[0], targets.shape[1]
    num_anchors = anchors.shape[0]

    ious = torch.zeros((num_batch, num_anchors, num_objects), device=anchors.device)
    zeros = torch.zeros(1, device=anchors.device)

    for idx, anchor in enumerate(anchors):
        inter_x1 = torch.maximum(anchor[0], targets[:, :, 0])
        inter_y1 = torch.maximum(anchor[1], targets[:, :, 1])
        inter_x2 = torch.minimum(anchor[2], targets[:, :, 2])
        inter_y2 = torch.minimum(anchor[3], targets[:, :, 3])

        inter_width = torch.maximum(zeros, inter_x2 - inter_x1)
        inter_height = torch.maximum(zeros, inter_y2 - inter_y1)

        inter = inter_width * inter_height
        union = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1]) + (targets[:, :, 2] - targets[:, :, 0]) * \
                (targets[:, :, 3] - targets[:, :, 1]) - inter

        iou = inter / union

        ious[:, idx, :] = iou

    return ious


def calc_reg_parameters(anchors: Tensor, targets: Tensor) -> Tensor:
    eps = torch.full((1,), torch.finfo().eps, device=anchors.device)

    anchors_width = torch.maximum(anchors[:, 2] - anchors[:, 0], eps)
    anchors_height = torch.maximum(anchors[:, 3] - anchors[:, 1], eps)
    anchors_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchors_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2

    targets_width = targets[:, :, 2] - targets[:, :, 0]
    targets_height = targets[:, :, 3] - targets[:, :, 1]
    targets_ctr_x = (targets[:, :, 0] + targets[:, :, 2]) / 2
    targets_ctr_y = (targets[:, :, 1] + targets[:, :, 3]) / 2

    dx = (targets_ctr_x - anchors_ctr_x) / anchors_width
    dy = (targets_ctr_y - anchors_ctr_y) / anchors_height
    dw = torch.log(torch.maximum(targets_width / anchors_width, eps))
    dh = torch.log(torch.maximum(targets_height / anchors_height, eps))

    reg_parameters = torch.stack((dx, dy, dw, dh), dim=2)

    return reg_parameters


def nms(rois: np.ndarray, scores: np.ndarray, nms_thresh: float) -> Tuple[np.ndarray, np.array]:
    order = scores.argsort().flip(dims=[0])

    keep_index = []

    while order.size(dim=0) > 0:
        i = order[0]
        keep_index.append(i)
        ious = bbox_iou(rois[i][np.newaxis, :], rois[order[1:]])
        inds = torch.where(ious <= nms_thresh)[1]
        order = order[inds + 1]

    return rois[keep_index]

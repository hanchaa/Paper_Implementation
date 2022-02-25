import numpy as np


def bbox_iou(anchors: np.ndarray, targets: np.ndarray) -> np.ndarray:
    len_anchors = anchors.shape[0]
    len_targets = targets.shape[0]

    ious = np.zeros((len_anchors, len_targets))

    for idx, anchor in enumerate(anchors):
        inter_x1 = np.maximum(anchor[0], targets[:, 0])
        inter_y1 = np.maximum(anchor[1], targets[:, 1])
        inter_x2 = np.minimum(anchor[2], targets[:, 2])
        inter_y2 = np.minimum(anchor[3], targets[:, 3])

        inter_width = np.maximum(0.0, inter_x2 - inter_x1)
        inter_height = np.maximum(0.0, inter_y2 - inter_y1)

        eps = np.finfo(np.float32).eps
        inter = inter_width * inter_height
        union = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1]) + (targets[:, 2] - targets[:, 0]) * \
                (targets[:, 3] - targets[:, 1]) - inter + eps

        iou = inter / union

        ious[idx] = iou

    return ious


def calc_reg_parameters(anchors: np.ndarray, targets: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).eps

    anchors_width = np.maximum(anchors[:, 2] - anchors[:, 0], eps)
    anchors_height = np.maximum(anchors[:, 3] - anchors[:, 1], eps)
    anchors_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchors_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2

    targets_width = targets[:, 2] - targets[:, 0]
    targets_height = targets[:, 3] - targets[:, 1]
    targets_ctr_x = (targets[:, 0] + targets[:, 2]) / 2
    targets_ctr_y = (targets[:, 1] + targets[:, 3]) / 2

    dx = (targets_ctr_x - anchors_ctr_x) / anchors_width
    dy = (targets_ctr_y - anchors_ctr_y) / anchors_height
    dw = np.log(targets_width / anchors_width)
    dh = np.log(targets_height / anchors_height)

    reg_parameters = np.stack((dx, dy, dw, dh), axis=1)
    return reg_parameters


def regress_roi(anchors: np.ndarray, reg_parameters: np.ndarray) -> np.ndarray:
    anchors_width = anchors[:, 2] - anchors[:, 0]
    anchors_height = anchors[:, 3] - anchors[:, 1]
    anchors_ctr_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchors_ctr_y = (anchors[:, 1] + anchors[:, 3]) / 2

    dx, dy, dw, dh = reg_parameters.T

    width = np.exp(dw) * anchors_width
    height = np.exp(dh) * anchors_height
    ctr_x = anchors_width * dx + anchors_ctr_x
    ctr_y = anchors_height * dy + anchors_ctr_y

    roi = np.zeros_like(anchors)
    roi[:, 0] = ctr_x - width * 0.5
    roi[:, 1] = ctr_y - height * 0.5
    roi[:, 2] = ctr_x + width * 0.5
    roi[:, 3] = ctr_y + width * 0.5

    return roi


def nms(rois: np.ndarray, scores: np.ndarray, nms_thresh: float) -> np.ndarray:
    order = scores.argsort()[::-1]

    keep_index = []

    while order.size > 0:
        i = order[0]
        keep_index.append(i)
        ious = bbox_iou(rois[i][np.newaxis, :], rois[order[1:]])
        inds = np.where(ious <= nms_thresh)[1]
        order = order[inds + 1]

    return rois[keep_index]

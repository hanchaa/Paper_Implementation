from typing import List, Tuple

from torch import Tensor

from utils import *


def create_anchors(anchor_scale: List[int], anchor_ratio: List[int], sub_sample_ratio: int,
                   feature_map_size: Tuple[int, int], image_size: Tuple[int, int]) -> np.ndarray:
    # create anchor template
    len_anchor_scale = len(anchor_scale)
    len_ratio = len(anchor_ratio)

    anchor_template = np.zeros((len_anchor_scale * len_ratio, 4))

    for idx, scale in enumerate(anchor_scale):
        h = scale * np.sqrt(anchor_ratio) * sub_sample_ratio
        w = scale / np.sqrt(anchor_ratio) * sub_sample_ratio

        x1 = -(w / 2)
        y1 = -(h / 2)
        x2 = w / 2
        y2 = h / 2

        anchor_template[idx * len_ratio:(idx + 1) * len_ratio, 0] = x1
        anchor_template[idx * len_ratio:(idx + 1) * len_ratio, 1] = y1
        anchor_template[idx * len_ratio:(idx + 1) * len_ratio, 2] = x2
        anchor_template[idx * len_ratio:(idx + 1) * len_ratio, 3] = y2

    # find anchor centers
    ctr = np.zeros((*feature_map_size, 2))
    ctr_x = np.arange(sub_sample_ratio // 2, image_size[0], sub_sample_ratio)
    ctr_y = np.arange(sub_sample_ratio // 2, image_size[1], sub_sample_ratio)

    for idx, y in enumerate(ctr_y):
        ctr[idx, :, 0] = ctr_x
        ctr[idx, :, 1] = y

    # create anchors
    anchors = np.zeros((*feature_map_size, *anchor_template.shape))

    for idx_y in range(feature_map_size[0]):
        for idx_x in range(feature_map_size[1]):
            anchors[idx_y, idx_x] = (ctr[idx_y, idx_x] + anchor_template.reshape((-1, 2, 2))).reshape((-1, 4))

    anchors = anchors.reshape((-1, 4))

    return anchors


def get_valid_anchor_indexes(anchors: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    # remove invalid anchors that cross the image border
    valid_index = np.where((anchors[:, 0] >= 0)
                           & (anchors[:, 1] >= 0)
                           & (anchors[:, 2] <= image_size[0])
                           & (anchors[:, 3] <= image_size[1]))[0]

    return valid_index


def create_anchors_label(anchors: np.ndarray, ious: np.ndarray, pos_iou_threshold: float,
                         neg_iou_threshold: float) -> np.ndarray:
    anchors_label = np.zeros((anchors.shape[0]))
    anchors_label.fill(-1)

    # first condition
    gt_max_iou = np.amax(ious, axis=0)
    gt_max_iou_anchor = np.where(ious == gt_max_iou)[0]

    anchors_label[gt_max_iou_anchor] = 1

    # second condition
    anchors_max_ious = np.amax(ious, axis=1)
    pos_iou_anchors = np.where(anchors_max_ious >= pos_iou_threshold)[0]
    neg_iou_anchors = np.where(anchors_max_ious < neg_iou_threshold)[0]

    anchors_label[pos_iou_anchors] = 1
    anchors_label[neg_iou_anchors] = 0

    return anchors_label


def sample_anchors(anchors_label: np.ndarray, num_sample: int, pos_ratio: float) -> np.ndarray:
    positive_anchor_indexes = np.where(anchors_label == 1)[0]
    negative_anchor_indexes = np.where(anchors_label == 0)[0]

    total_num_pos = len(positive_anchor_indexes)

    num_pos_sample = num_sample * pos_ratio if total_num_pos > num_sample * pos_ratio else total_num_pos
    num_neg_sample = num_sample - num_pos_sample

    if total_num_pos > num_sample * pos_ratio:
        disable_index = np.random.choice(positive_anchor_indexes, size=total_num_pos - num_pos_sample, replace=False)
        anchors_label[disable_index] = -1

    disable_index = np.random.choice(negative_anchor_indexes, size=len(negative_anchor_indexes) - num_neg_sample,
                                     replace=False)
    anchors_label[disable_index] = -1

    return anchors_label


def calc_anchors_reg_parameters(anchors: np.ndarray, bboxes: np.ndarray, ious: np.ndarray) -> np.ndarray:
    argmax_ious = np.argmax(ious, axis=1)
    max_iou_boxes = bboxes[argmax_ious]

    reg_parameters = calc_reg_parameters(anchors, max_iou_boxes)

    return reg_parameters


def create_target_anchors(num_anchors: int, num_sample: int, valid_anchors: np.ndarray, valid_anchor_indexes: np.ndarray,
                          bboxes: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ious = bbox_iou(valid_anchors, bboxes.to("cpu").numpy())

    valid_anchors_label = create_anchors_label(valid_anchors, ious, 0.7, 0.3)
    valid_anchors_label = sample_anchors(valid_anchors_label, num_sample, 0.5)

    target_anchors_label = np.empty(num_anchors, dtype=np.int32)
    target_anchors_label.fill(-1)
    target_anchors_label[valid_anchor_indexes] = valid_anchors_label

    valid_anchors_reg_parameter = calc_anchors_reg_parameters(valid_anchors, bboxes.to("cpu").numpy(), ious)

    target_anchors_reg_parameter = np.zeros((num_anchors, 4), dtype=np.float32)
    target_anchors_reg_parameter[valid_anchor_indexes] = valid_anchors_reg_parameter

    return target_anchors_label, target_anchors_reg_parameter, ious

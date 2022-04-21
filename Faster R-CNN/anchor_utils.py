from typing import List

from utils import *


def create_anchors(anchor_scale: List[int], anchor_ratio: List[int], sub_sample_ratio: int,
                   image_size: Tuple[int, int], device) -> Tensor:
    feature_map_size = (image_size[0] // sub_sample_ratio, image_size[1] // sub_sample_ratio)

    # create anchor template
    len_anchor_scale = len(anchor_scale)
    len_ratio = len(anchor_ratio)

    anchor_template = torch.zeros((len_anchor_scale * len_ratio, 4), device=device)

    for idx, scale in enumerate(anchor_scale):
        h = scale * torch.sqrt(Tensor(anchor_ratio)) * sub_sample_ratio
        w = scale / torch.sqrt(Tensor(anchor_ratio)) * sub_sample_ratio

        x1 = -(w / 2)
        y1 = -(h / 2)
        x2 = w / 2
        y2 = h / 2

        anchor_template[idx * len_ratio:(idx + 1) * len_ratio, 0] = x1
        anchor_template[idx * len_ratio:(idx + 1) * len_ratio, 1] = y1
        anchor_template[idx * len_ratio:(idx + 1) * len_ratio, 2] = x2
        anchor_template[idx * len_ratio:(idx + 1) * len_ratio, 3] = y2

    # find anchor centers
    ctr = torch.zeros((*feature_map_size, 2), device=device)
    ctr_x = torch.arange(sub_sample_ratio // 2, image_size[0], sub_sample_ratio, device=device)
    ctr_y = torch.arange(sub_sample_ratio // 2, image_size[1], sub_sample_ratio, device=device)

    for idx, y in enumerate(ctr_y):
        ctr[idx, :, 0] = ctr_x
        ctr[idx, :, 1] = y

    # create anchors
    anchors = torch.zeros((*feature_map_size, *anchor_template.shape)).to(device)

    for idx_y in range(feature_map_size[0]):
        for idx_x in range(feature_map_size[1]):
            anchors[idx_y, idx_x] = (ctr[idx_y, idx_x] + anchor_template.reshape((-1, 2, 2))).reshape((-1, 4))

    anchors = anchors.reshape((-1, 4))

    return anchors


def get_valid_anchor_indexes(anchors: Tensor, image_size: Tuple[int, int]) -> Tensor:
    # remove invalid anchors that cross the image border
    valid_index = torch.where((anchors[:, 0] >= 0)
                              & (anchors[:, 1] >= 0)
                              & (anchors[:, 2] <= image_size[0])
                              & (anchors[:, 3] <= image_size[1]))[0]

    return valid_index


def create_anchors_label(anchors: Tensor, ious: Tensor, pos_iou_threshold: float,
                         neg_iou_threshold: float) -> Tensor:
    anchors_label = torch.full((ious.shape[0], anchors.shape[0],), -1, device=anchors.device, dtype=torch.int8)

    # first condition
    gt_max_iou_anchor = torch.argmax(ious, dim=1)

    for batch in range(ious.shape[0]):
        anchors_label[batch, gt_max_iou_anchor[batch]] = 1

    # second condition
    anchors_max_ious = torch.amax(ious, dim=2)
    pos_iou_anchors = torch.where(anchors_max_ious >= pos_iou_threshold, 1, 0).bool()
    neg_iou_anchors = torch.where(anchors_max_ious < neg_iou_threshold, 1, 0).bool()

    anchors_label[pos_iou_anchors] = 1
    anchors_label[neg_iou_anchors] = 0

    return anchors_label


def sample_anchors(anchors_label: Tensor, num_sample: int, pos_ratio: float) -> Tensor:
    positive_anchor_indexes = torch.stack(torch.where(anchors_label == 1))
    negative_anchor_indexes = torch.stack(torch.where(anchors_label == 0))

    total_num_pos = positive_anchor_indexes.shape[1]

    num_pos_sample = int(num_sample * pos_ratio) if total_num_pos > num_sample * pos_ratio else total_num_pos
    num_neg_sample = num_sample - num_pos_sample

    if total_num_pos > num_sample * pos_ratio:
        sample = torch.multinomial(torch.ones_like(positive_anchor_indexes[0]).float(),
                                   num_samples=total_num_pos - num_pos_sample)
        disable_index = positive_anchor_indexes[:, sample]
        anchors_label[disable_index[0], disable_index[1]] = -1

    sample = torch.multinomial(torch.ones_like(negative_anchor_indexes[0]).float(),
                               num_samples=negative_anchor_indexes.shape[1] - num_neg_sample)
    disable_index = negative_anchor_indexes[:, sample]
    anchors_label[disable_index[0], disable_index[1]] = -1

    return anchors_label


def calc_anchors_reg_parameters(anchors: Tensor, bboxes: Tensor, ious: Tensor) -> Tensor:
    argmax_ious = torch.argmax(ious, dim=-1)

    max_iou_boxes = []
    for batch in range(bboxes.shape[0]):
        max_iou_boxes.append(bboxes[batch, argmax_ious[batch]])
    max_iou_boxes = torch.stack(max_iou_boxes)

    reg_parameters = calc_reg_parameters(anchors, max_iou_boxes)

    return reg_parameters


def create_target_anchors(num_anchors: int, num_sample: int, valid_anchors: Tensor,
                          valid_anchor_indexes: Tensor, bboxes: Tensor) \
        -> Tuple[Tensor, Tensor, Tensor]:
    ious = bbox_iou(valid_anchors, bboxes)

    valid_anchors_label = create_anchors_label(valid_anchors, ious, 0.7, 0.3)
    valid_anchors_label = sample_anchors(valid_anchors_label, num_sample, 0.5)

    target_anchors_label = torch.full((valid_anchors_label.shape[0], num_anchors,), -1, dtype=torch.int8, device=valid_anchors.device)
    target_anchors_label[:, valid_anchor_indexes] = valid_anchors_label

    valid_anchors_reg_parameter = calc_anchors_reg_parameters(valid_anchors, bboxes, ious)

    target_anchors_reg_parameter = torch.zeros((valid_anchors_label.shape[0], num_anchors, 4), dtype=torch.float32, device=valid_anchors.device)
    target_anchors_reg_parameter[:, valid_anchor_indexes] = valid_anchors_reg_parameter

    return target_anchors_label, target_anchors_reg_parameter, ious

from typing import Any

import torch
from torch import nn

from anchor_utils import *

rpn_cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
rpn_reg_loss_fn = nn.SmoothL1Loss(reduction="sum")


def calc_rpn_loss(predicted_anchors_class_score: Tensor,
                  predicted_anchors_reg_parameter: Tensor,
                  valid_anchors: np.ndarray,
                  valid_anchor_indexes: np.ndarray,
                  bboxes: Tensor,
                  num_anchors: int,
                  num_anchor_sample: int,
                  rpn_lambda: float) -> Tuple[Any, np.ndarray]:
    target_anchors_label, target_anchors_reg_parameter, ious = create_target_anchors(num_anchors, num_anchor_sample,
                                                                                     valid_anchors,
                                                                                     valid_anchor_indexes, bboxes)

    # loss of rpn
    gt_rpn_label = torch.from_numpy(target_anchors_label)
    rpn_score = predicted_anchors_class_score[0]
    rpn_cls_loss = rpn_cls_loss_fn(rpn_score, gt_rpn_label.long())

    mask = gt_rpn_label > 0
    gt_rpn_reg_parameter = torch.from_numpy(target_anchors_reg_parameter)[mask]
    rpn_reg_parameter = predicted_anchors_reg_parameter[0][mask]
    rpn_reg_loss = rpn_reg_loss_fn(rpn_reg_parameter, gt_rpn_reg_parameter)

    num_reg = mask.float().sum()
    rpn_loss = rpn_cls_loss / num_anchor_sample + (rpn_lambda / num_reg) * rpn_reg_loss

    return rpn_loss, ious

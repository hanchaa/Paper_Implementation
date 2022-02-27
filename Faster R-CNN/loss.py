from torch import nn

from anchor_utils import *


class RPNLoss:
    def __init__(self, anchors: Tensor, image_size: Tuple[int, int], num_anchor_samples, rpn_lambda):
        self.anchors = anchors
        self.valid_anchor_indexes = get_valid_anchor_indexes(anchors, image_size)
        self.valid_anchors = anchors[self.valid_anchor_indexes]

        self.num_anchor_samples = num_anchor_samples
        self.rpn_lambda = rpn_lambda

        self.rpn_cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        self.rpn_reg_loss_fn = nn.SmoothL1Loss(reduction="sum")

    def __call__(self,
                 predicted_anchors_location: Tensor,
                 predicted_anchors_class_score: Tensor,
                 bboxes: Tensor,
                 device: str):
        gt_rpn_label, gt_rpn_reg_parameter, ious = create_target_anchors(len(self.anchors),
                                                                         self.num_anchor_samples,
                                                                         self.valid_anchors,
                                                                         self.valid_anchor_indexes,
                                                                         bboxes)

        # loss of rpn
        rpn_score = predicted_anchors_class_score[0]
        rpn_cls_loss = self.rpn_cls_loss_fn(rpn_score, gt_rpn_label.long())

        mask = gt_rpn_label > 0
        gt_rpn_reg_parameter = gt_rpn_reg_parameter[mask]
        rpn_reg_parameter = calc_reg_parameters(self.anchors, predicted_anchors_location[0])
        rpn_reg_parameter = rpn_reg_parameter[mask]
        rpn_reg_loss = self.rpn_reg_loss_fn(rpn_reg_parameter, gt_rpn_reg_parameter)

        num_reg = mask.float().sum()
        rpn_loss = rpn_cls_loss / self.num_anchor_samples + (self.rpn_lambda / num_reg) * rpn_reg_loss

        return rpn_loss

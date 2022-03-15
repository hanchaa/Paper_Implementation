from torch import nn

from anchor_utils import *


class RPNLoss:
    def __init__(self, anchors: Tensor, image_size: Tuple[int, int], num_anchor_samples: int, rpn_lambda: float):
        self.anchors = anchors
        self.valid_anchor_indexes = get_valid_anchor_indexes(anchors, image_size)
        self.valid_anchors = anchors[self.valid_anchor_indexes]

        self.num_anchor_samples = num_anchor_samples
        self.rpn_lambda = rpn_lambda

        self.cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.reg_loss_fn = nn.SmoothL1Loss(reduction="sum")

    def __call__(self,
                 predicted_anchors_location: Tensor,
                 predicted_anchors_class_score: Tensor,
                 bboxes: Tensor):
        gt_rpn_label, gt_rpn_reg_parameter, ious = create_target_anchors(len(self.anchors),
                                                                         self.num_anchor_samples,
                                                                         self.valid_anchors,
                                                                         self.valid_anchor_indexes,
                                                                         bboxes)

        # loss of rpn
        rpn_score = predicted_anchors_class_score[0]
        rpn_cls_loss = self.cls_loss_fn(rpn_score, gt_rpn_label.long())

        mask = gt_rpn_label > 0
        gt_rpn_reg_parameter = gt_rpn_reg_parameter[mask]
        rpn_reg_parameter = calc_reg_parameters(self.anchors, predicted_anchors_location[0])
        rpn_reg_parameter = rpn_reg_parameter[mask]
        rpn_reg_loss = self.reg_loss_fn(rpn_reg_parameter, gt_rpn_reg_parameter)

        num_reg = mask.float().sum()
        rpn_loss = rpn_cls_loss + (self.rpn_lambda / num_reg) * rpn_reg_loss

        return rpn_loss


class FastRCNNLoss:
    def __init__(self, roi_lambda: float):
        self.fast_rcnn_lambda = roi_lambda

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss(reduction="sum")

    def __call__(self, predicted_locations: Tensor, predicted_class_scores: Tensor, targets_location: Tensor,
                 targets_label: Tensor):
        fast_rcnn_cls_loss = self.cls_loss_fn(predicted_class_scores, targets_label)

        num_roi = predicted_locations.shape[0]
        predicted_locations = predicted_locations[torch.arange(num_roi), targets_label]

        mask = targets_label > 0
        masked_predicted_locations = predicted_locations[mask]
        masked_targets_location = targets_location[mask]
        fast_rcnn_reg_loss = self.reg_loss_fn(masked_predicted_locations, masked_targets_location)

        num_reg = mask.float().sum()
        fast_rcnn_loss = fast_rcnn_cls_loss + (self.fast_rcnn_lambda / num_reg) * fast_rcnn_reg_loss

        return fast_rcnn_loss

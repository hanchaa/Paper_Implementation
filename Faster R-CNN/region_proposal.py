from typing import Tuple

import torch

from utils import nms


class RegionProposal:
    def __init__(self,
                 img_size: Tuple[int, int],
                 num_train_pre_nms: int = 12000,
                 num_train_post_nms: int = 2000,
                 num_test_pre_nms: int = 6000,
                 num_test_post_nms: int = 300,
                 min_roi_size: int = 16, nms_threshold: float = 0.7):
        self.img_size = img_size
        self.num_train_pre_nms = num_train_pre_nms
        self.num_train_post_nms = num_train_post_nms
        self.num_test_pre_nms = num_test_pre_nms
        self.num_test_post_nms = num_test_post_nms
        self.min_roi_size = min_roi_size
        self.nms_threshold = nms_threshold

    def __call__(self, predicted_anchors_location, predicted_anchors_class_score, is_training):
        rois = predicted_anchors_location[0].detach()
        rois[:, 0:4:2] = torch.clip(rois[:, 0:4:2], min=0, max=self.img_size[1])
        rois[:, 1:4:2] = torch.clip(rois[:, 1:4:2], min=0, max=self.img_size[0])

        rois_w = rois[:, 2] - rois[:, 0]
        rois_h = rois[:, 3] - rois[:, 1]

        valid_index = torch.where((rois_w > self.min_roi_size) & (rois_h > self.min_roi_size))[0]
        rois = rois[valid_index]

        roi_scores = predicted_anchors_class_score[0, :, 1][valid_index].detach()
        score_order = roi_scores.argsort().flip(dims=[0])

        num_pre_nms = self.num_train_pre_nms if is_training else self.num_test_pre_nms
        num_post_nms = self.num_train_post_nms if is_training else self.num_test_post_nms

        pre_nms_score_order = score_order[:num_pre_nms]
        pre_nms_rois = rois[pre_nms_score_order]
        pre_nms_roi_scores = roi_scores[pre_nms_score_order]

        post_nms_rois = nms(pre_nms_rois, pre_nms_roi_scores, self.nms_threshold)[:num_post_nms]

        return post_nms_rois

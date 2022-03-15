import torch
from torch import Tensor

from utils import bbox_iou


class RoiSampler:
    def __init__(self, num_sample: int = 128, pos_ratio: float = 0.25, pos_iou_thresh: float = 0.5,
                 neg_iou_thresh: float = 0.5):
        self.num_sample = num_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

    def __call__(self, rois: Tensor, bboxes: Tensor, labels: Tensor):
        ious = bbox_iou(rois, bboxes)

        bbox_assignments = ious.argmax(dim=1)
        roi_max_ious = ious.max(dim=1)

        target_locations = bboxes[bbox_assignments]
        roi_target_labels = labels[bbox_assignments]

        total_num_pos = len(torch.where(roi_max_ious >= self.pos_iou_thresh)[0])
        num_pos_sample = self.num_sample * self.pos_ratio if total_num_pos >= self.num_sample * self.pos_ratio else total_num_pos
        num_neg_sample = self.num_sample - num_pos_sample

        pos_indexes = torch.where(roi_max_ious >= self.pos_iou_thresh)[0]
        pos_indexes = pos_indexes[torch.multinomial(torch.ones_like(pos_indexes).float(), num_samples=num_pos_sample)]

        neg_indexes = torch.where(roi_max_ious < self.neg_iou_thresh)[0]
        neg_indexes = neg_indexes[torch.multinomial(torch.ones_like(neg_indexes).float(), num_samples=num_neg_sample)]

        keep_indexes = torch.cat((pos_indexes, neg_indexes))

        sampled_target_locations = target_locations[keep_indexes]
        sampled_target_labels = roi_target_labels[keep_indexes] + 1
        sampled_target_labels[len(pos_indexes):] = 0
        sampled_rois = rois[keep_indexes]

        return sampled_rois, sampled_target_locations, sampled_target_labels

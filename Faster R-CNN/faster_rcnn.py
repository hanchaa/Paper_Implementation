from torch import nn

from anchor_utils import create_anchors


class FasterRCNN(nn.Module):
    def __init__(self, anchors_scale, anchors_ratio, sub_sample_ratio, image_size, feature_extractor, rpn,
                 region_proposal=None, roi_sampler=None, fast_rcnn=None, device="cpu"):
        super().__init__()

        self.anchors = create_anchors(anchors_scale, anchors_ratio, sub_sample_ratio, image_size, device)

        self.feature_extractor = feature_extractor
        self.rpn = rpn
        self.region_proposal = region_proposal
        self.roi_sampler = roi_sampler
        self.fast_rcnn = fast_rcnn

    def forward(self, x, bboxes, labels):
        feature_map = self.feature_extractor(x)

        predicted_anchors_location, predicted_anchors_class_score = self.rpn(feature_map)

        if self.region_proposal is not None:
            rois = self.region_proposal(predicted_anchors_location, predicted_anchors_class_score, self.training)
            rois, targets_location, targets_label = self.roi_sampler(rois, bboxes, labels)
            predicted_locations, predicted_class_scores = self.fast_rcnn(rois, feature_map)

            return [predicted_anchors_location, predicted_anchors_class_score, predicted_locations, predicted_class_scores, targets_location, targets_label]

        return [predicted_anchors_location, predicted_anchors_class_score]

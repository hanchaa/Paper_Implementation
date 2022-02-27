from torch import nn

from anchor_utils import create_anchors


class FasterRCNN(nn.Module):
    def __init__(self, anchors_scale, anchors_ratio, sub_sample_ratio, image_size, feature_extractor, rpn,
                 region_proposal, device):
        super().__init__()

        self.anchors = create_anchors(anchors_scale, anchors_ratio, sub_sample_ratio, image_size, device)

        self.feature_extractor = feature_extractor
        self.rpn = rpn
        self.region_proposal = region_proposal

    def forward(self, x):
        feature_map = self.feature_extractor(x)

        predicted_anchors_location, predicted_anchors_class_score = self.rpn(feature_map)

        # self.region_proposal(predicted_anchors_reg_parameter, predicted_anchors_class_score, self.training)

        return predicted_anchors_location, predicted_anchors_class_score

from torch import nn


class FasterRCNN(nn.Module):
    def __init__(self, feature_extractor, rpn):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.rpn = rpn

    def forward(self, x):
        feature_map = self.feature_extractor(x)

        predicted_anchors_reg_parameter, predicted_anchors_class_score = self.rpn(feature_map)

        return predicted_anchors_reg_parameter, predicted_anchors_class_score

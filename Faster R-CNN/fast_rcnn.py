from typing import Tuple

import torch
from einops import rearrange
from torch import nn


class FastRCNN(nn.Module):
    def __init__(self, size: Tuple[int, int], feature_map_dim: int, sub_sample_ratio: int, hidden_size: int,
                 num_classes: int):
        super().__init__()

        self.sub_sample_ratio = sub_sample_ratio

        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(size)
        self.roi_head = nn.Sequential(
            nn.Linear(size[0] * size[1] * feature_map_dim, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self.location = nn.Linear(hidden_size, num_classes * 4)
        self.score = nn.Linear(hidden_size, num_classes)

    def forward(self, rois, feature_map):
        rois = rois / self.sub_sample_ratio
        rois = rois.long()

        roi_features = []
        for roi in rois:
            roi_feature = feature_map[..., roi[1]:roi[3] + 1, roi[0]:roi[2] + 1]
            roi_features.append(roi_feature)

        roi_features = torch.stack(roi_features, dim=0)
        roi_features = self.adaptive_max_pool(roi_features)

        roi_features = rearrange(roi_features, "b n h w -> b (n h w)")

        output = self.roi_head(roi_features)
        predicted_locations = self.location(output)
        predicted_locations = rearrange(predicted_locations, "b (c n) -> b c n", n=4)
        predicted_class_scores = self.score(output)

        return predicted_locations, predicted_class_scores

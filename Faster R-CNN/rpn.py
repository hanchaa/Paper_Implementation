from einops import rearrange
from torch import nn


class RPN(nn.Module):
    def __init__(self, in_channel: int = 512, mid_channel: int = 512, num_anchors: int = 9):
        super().__init__()

        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1)
        self.reg_layer = nn.Conv2d(mid_channel, num_anchors * 4, kernel_size=1, stride=1)
        self.cls_layer = nn.Conv2d(mid_channel, num_anchors * 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)

        predicted_anchors_location = rearrange(self.reg_layer(x), "b (n e) h w -> b (h w n) e", n=self.num_anchors)
        predicted_anchors_class_score = rearrange(self.cls_layer(x), "b (n e) h w -> b (h w n) e", n=self.num_anchors)

        return predicted_anchors_location, predicted_anchors_class_score

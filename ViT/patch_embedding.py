import torch

from torch import nn, Tensor
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_size: int = 768, img_size: int = 224):
        super().__init__()

        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.pos_embedding = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, embedding_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape

        x = self.projection(x)

        cls_token = repeat(self.cls_token, "() n e -> b n e", b=b)
        x = torch.concat((cls_token, x), dim=1)

        out = x + self.pos_embedding

        return out

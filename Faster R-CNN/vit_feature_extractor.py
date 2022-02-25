import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "\\ViT")

from torch import nn
from einops import rearrange
from ViT.patch_embedding import PatchEmbedding
from ViT.transformer_encoder import TransformerEncoder


class ViTFeatureExtractor(nn.Module):
    def __init__(self, in_channel: int = 3, patch_size: int = 16, embedding_size: int = 768, image_size: int = 800,
                 depth: int = 12, num_heads: int = 12, mlp_expansion: int = 4):
        super().__init__()

        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.image_size = image_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_expansion = mlp_expansion

        self.patch_embedding = PatchEmbedding(in_channel, patch_size, embedding_size, image_size)
        self.transformer_encoder = TransformerEncoder(depth, embedding_size, num_heads, mlp_expansion)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)[:, 1:, :]
        x = rearrange(x, "b (h w) e -> b e h w", h=self.image_size // self.patch_size,
                      w=self.image_size // self.patch_size)

        return x

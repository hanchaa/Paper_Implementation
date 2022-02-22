from torch import nn

from classification_head import ClassificationHead
from patch_embedding import PatchEmbedding
from transformer_encoder import TransformerEncoder


class ViT(nn.Sequential):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embedding_size: int = 768, img_size: int = 224,
                 depth: int = 12, num_heads: int = 8, mlp_expansion: int = 4, num_classes: int = 1000):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, embedding_size, img_size),
            TransformerEncoder(depth, embedding_size, num_heads, mlp_expansion),
            ClassificationHead(embedding_size, num_classes)
        )

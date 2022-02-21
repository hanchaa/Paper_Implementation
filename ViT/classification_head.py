from einops.layers.torch import Reduce
from torch import nn


class ClassificationHead(nn.Sequential):
    def __init__(self, embedding_size: int = 768, num_classes: int = 1000):
        super().__init__(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, num_classes)
        )

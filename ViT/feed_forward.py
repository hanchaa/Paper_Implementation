from torch import nn


class FeedForward(nn.Sequential):
    def __init__(self, embedding_size: int, expansion: int = 4):
        super().__init__(
            nn.Linear(embedding_size, embedding_size * expansion),
            nn.GELU(),
            nn.Linear(embedding_size * expansion, embedding_size),
        )

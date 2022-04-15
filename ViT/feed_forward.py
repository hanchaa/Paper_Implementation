from torch import nn


class FeedForward(nn.Sequential):
    def __init__(self, embedding_size: int, expansion: int = 4, dropout: float = 0):
        super().__init__(
            nn.Linear(embedding_size, embedding_size * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size * expansion, embedding_size),
            nn.Dropout(dropout)
        )

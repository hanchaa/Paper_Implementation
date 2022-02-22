from torch import nn

from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention
from residual_add import ResidualAdd


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, embedding_size: int = 768, num_heads: int = 8, mlp_expansion: int = 4):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_size),
                MultiHeadAttention(embedding_size, num_heads)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(embedding_size),
                FeedForward(embedding_size, mlp_expansion)
            ))
        )

from torch import nn

from transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, embedding_size: int = 768, num_heads: int = 8, mlp_expansion: int = 4,
                 attention_dropout: float = 0, mlp_dropout: float = 0):
        super().__init__(
            *[TransformerEncoderBlock(embedding_size, num_heads, mlp_expansion, attention_dropout, mlp_dropout) for _ in
              range(depth)]
        )
